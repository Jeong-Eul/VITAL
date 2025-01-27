"""Here is an initial version of the proposed EHRTimeLLM model. The scripts will be further refined in the future, after paper acceptance. """

import os
import random
# import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
from model_ehr import *
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
from multiprocessing import freeze_support
import torch.distributed as dist
import atexit
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Process group destroyed.")
        
def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        print(f"Process group initialized on rank {dist.get_rank()}.")
        
if __name__ == '__main__':
    
    freeze_support()
    torch.manual_seed(1)
    atexit.register(cleanup)
    initialize_distributed()
    
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100000"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['WANDB_SILENT']="true"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'eICU', 'PAM']) #
    parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
    parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
    parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training') #
    parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype==random; otherwise, set as no_removal') #
    parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                        help='use this only with P12 dataset (mortality or length of stay)')


    # Model configuration
    parser.add_argument('--enc_in', type=int, default=34, help='encoder input size')
    parser.add_argument('--num_tokens', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT, MAMBA
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768, MAMBA:768
    parser.add_argument('--llm_layers', type=int, default=12)
    
    wandb = False
    
    args, unknown = parser.parse_known_args()
    
    torch.manual_seed(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    arch = 'ehrtimellm'
    model_path = '../models/'
    args.batch_size = 32
    dataset = args.dataset
    print('Dataset used: ', dataset)

    if dataset == 'P12':
        base_path = '../P12data'
    elif dataset == 'P19':
        base_path = '../P19data'
    elif dataset == 'PAM':
        base_path = '../PAMdata'

    baseline = False  # always False for Raindrop
    split = args.splittype  # possible values: 'random', 'age', 'gender'
    reverse = args.reverse  # False or True
    feature_removal_level = args.feature_removal_level  # 'set', 'sample'
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=4, deepspeed_plugin=None)
    
    def one_hot(y_):
        y_ = y_.reshape(len(y_))

        y_ = [int(x) for x in y_]
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]
    
    def split_randomly(num_var, seed=2):
        random.seed(seed)
        numbers = list(range(num_var))
        random.shuffle(numbers)

        list1 = numbers[:8]
        list2 = numbers[8:]

        return list1, list2

    print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level',
        args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level)

    """While missing_ratio >0, feature_removal_level is automatically used"""
    if args.withmissingratio == True:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    print('missing ratio list', missing_ratios)

    sensor_wise_mask = False

    for missing_ratio in missing_ratios:
        num_epochs = 20
        learning_rate = 0.001

        if dataset == 'P12':
            args.d_static = 9
            args.enc_in = 36
            args.static_info = 1
            args.vital = [8, 14, 21, 22, 30] # 순서대로 DBP, HR, NDBP, NMAP, SBP |  NSBP, MAP, RR은 EDA 결과 lab 만큼 희소해서 제외
            args.lab = list(set(np.arange(args.enc_in)) - set(args.vital))
            
        elif dataset == 'P19':
            args.d_static = 6
            args.enc_in = 34
            args.static_info = 1
            args.vital = [0, 1, 3, 4, 5]
            args.lab = list(set(np.arange(args.enc_in)) - set(args.vital))
            
        elif dataset == 'PAM':
            args.d_static = 0
            args.enc_in = 17
            args.static_info = None
            args.vital, args.lab = split_randomly(args.enc_in, seed=42)

        if dataset == 'P12':
            args.seq_len = 215
            args.n_classes = 2
        elif dataset == 'P19':
            args.seq_len = 60
            args.n_classes = 2
        elif dataset == 'PAM':
            args.seq_len = 600
            args.n_classes = 8

        n_runs = 1
        n_splits = 5
        subset = False

        acc_arr = np.zeros((n_splits, n_runs))
        auprc_arr = np.zeros((n_splits, n_runs))
        auroc_arr = np.zeros((n_splits, n_runs))
        precision_arr = np.zeros((n_splits, n_runs))
        recall_arr = np.zeros((n_splits, n_runs))
        F1_arr = np.zeros((n_splits, n_runs))
        # for k in range(n_splits):
        custom = [0, 1, 2, 3, 4]
        for k in custom:
            split_idx = k + 1
            
            print('Split id: %d' % split_idx)
            if dataset == 'P12':
                if subset == True:
                    split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
                else:
                    split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
            elif dataset == 'P19':
                split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
            elif dataset == 'PAM':
                split_path = '/splits/PAMAP2_split_' + str(split_idx) + '.npy'
                
            if wandb:
                # wandb.login(key=str('0126f71b25a3ecd1e32ed0a83047073475ee9cea'))
                # config = wandb.config
                wandb.init(name=f'P19-EXP'+ str(split_idx),
                           project='EHRTimeLLM-refined', 
                           config={'Learning Rate':learning_rate, "LLM": args.llm_model, "d_ff": args.d_ff, "d_model": args.d_model,"Heads": args.n_heads})

            # prepare the data:
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split, reverse=reverse,
                                                                    baseline=baseline, dataset=dataset,
                                                                    predictive_label=args.predictive_label)
            
            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

            if dataset == 'P12' or dataset == 'P19':
                T, F = Ptrain[0]['arr'].shape
                D = len(Ptrain[0]['extended_static'])

                Ptrain_tensor = np.zeros((len(Ptrain), T, F))
                Ptrain_static_tensor = np.zeros((len(Ptrain), D))

                for i in range(len(Ptrain)):
                    Ptrain_tensor[i] = Ptrain[i]['arr']
                    Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

                mf, stdf = getStats(Ptrain_tensor)
                ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

                Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, train_paddimg_mask = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                            stdf, ms, ss)
                Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, valid_paddimg_mask = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
                Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, test_paddimg_mask = tensorize_normalize(Ptest, ytest, mf, stdf, ms,
                                                                                                ss)
            elif dataset == 'PAM':
                T, F = Ptrain[0].shape
                D = 1

                Ptrain_tensor = Ptrain
                Ptrain_static_tensor = np.zeros((len(Ptrain), D))

                mf, stdf = getStats(Ptrain)
                Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, train_paddimg_mask = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
                Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, valid_paddimg_mask = tensorize_normalize_other(Pval, yval, mf, stdf)
                Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, test_paddimg_mask = tensorize_normalize_other(Ptest, ytest, mf, stdf)

            # remove part of variables in validation and test set
            if missing_ratio > 0:
                num_all_features =int(Pval_tensor.shape[2] / 2)
                num_missing_features = round(missing_ratio * num_all_features)
                if feature_removal_level == 'sample':
                    for i, patient in enumerate(Pval_tensor):
                        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                        patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)
                        Pval_tensor[i] = patient
                    for i, patient in enumerate(Ptest_tensor):
                        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                        patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)
                        Ptest_tensor[i] = patient
                elif feature_removal_level == 'set':
                    density_score_indices = np.load('./baselines/saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
                    idx = density_score_indices[:num_missing_features].astype(int)
                    Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)
                    Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)

            # Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2) # Time, N_patient, Nvariable + Nmask
            # Pval_tensor = Pval_tensor.permute(1, 0, 2)
            # Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

            Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0) # Time, N_patient
            Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
            Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

            for m in range(n_runs):
                print('- - Run %d - -' % (m + 1))

                model = Ehrtimellm(args)
     
                trained_parameters = []
                for p in model.parameters():
                    if p.requires_grad is True:
                        trained_parameters.append(p)

                criterion = torch.nn.CrossEntropyLoss().to(accelerator.device)
                optimizer = torch.optim.Adam(trained_parameters, lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                    patience=1, threshold=0.0001, threshold_mode='rel',
                                                                    cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
                
                model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

                idx_0 = np.where(ytrain == 0)[0]
                idx_1 = np.where(ytrain == 1)[0]

                if dataset == 'P12' or dataset == 'P19':
                    strategy = 2
                elif dataset == 'PAM':
                    strategy = 3

                n0, n1 = len(idx_0), len(idx_1)
                expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
                expanded_n1 = len(expanded_idx_1)

                
                batch_size = args.batch_size
                if strategy == 1:
                    n_batches = 10
                elif strategy == 2:
                    K0 = n0 // int(batch_size / 2)
                    K1 = expanded_n1 // int(batch_size / 2)
                    n_batches = np.min([K0, K1])
                elif strategy == 3:
                    n_batches = 30

                best_aupr_val = best_auc_val = 0.0
                best_loss_val = 100.0
                patience = 0
                print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))

                start = time.time()
                if wandb:
                    wandb.watch(model)
                for epoch in range(num_epochs):
                    model.train()
                    start_epoch = time.time()
                    if strategy == 2:
                        np.random.shuffle(expanded_idx_1)
                        I1 = expanded_idx_1
                        np.random.shuffle(idx_0)
                        I0 = idx_0

                    for n in range(n_batches):
                        if strategy == 1:
                            idx = random_sample(idx_0, idx_1, batch_size)
                        elif strategy == 2:
                            idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                            idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                            idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                        elif strategy == 3:
                            idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                            # idx = random_sample_8(ytrain, batch_size)   # to balance dataset

                        if dataset == 'P12' or dataset == 'P19':
                            P, Ptime, Pstatic, y, mask = Ptrain_tensor[idx, :, :int(Ptrain_tensor.shape[2] / 2)].to(accelerator.device), Ptrain_time_tensor[:, idx].to(accelerator.device), \
                                                Ptrain_static_tensor[idx].to(accelerator.device), ytrain_tensor[idx].to(accelerator.device), train_paddimg_mask[idx].to(accelerator.device)
                        elif dataset == 'PAM':
                            
                            P, Ptime, Pstatic, y, mask = Ptrain_tensor[idx, :, :int(Ptrain_tensor.shape[2] / 2)].to(accelerator.device), Ptrain_time_tensor[:, idx].to(accelerator.device), \
                                                None, ytrain_tensor[idx].to(accelerator.device), train_paddimg_mask[idx].to(accelerator.device)

                        real_time = torch.sum(Ptime > 0, dim=0)
                        
                        with accelerator.accumulate(model):

                            outputs = model.forward(P, Ptime, real_time, Pstatic, mask)
                            loss = criterion(outputs, y)
                        
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                    end_epoch = time.time()
                    accelerator.wait_for_everyone()
                    
                
                    if dataset == 'P12' or dataset == 'P19' :
                        train_probs = torch.squeeze(torch.sigmoid(outputs))
                        train_probs = train_probs.cpu().detach().numpy()
                        train_y = y.cpu().detach().numpy()
                        train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                        train_auprc = average_precision_score(train_y, train_probs[:, 1])
                    elif dataset == 'PAM':
                        train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))
                        train_probs = train_probs.cpu().detach().numpy()
                        train_y = y.cpu().detach().numpy()
                        
                        try:
                            if len(np.unique(train_y)) == 1:
                                print("Only one class present in train_y. Skipping AUC calculation.")
                                train_auroc = float(0.5)
                            else:
                                try:
                                    train_auroc = roc_auc_score(one_hot(train_y), train_probs)
                                except: 
                                    train_auroc = float(0.5)  # 기본값 설정
                        except ValueError as e:
                            print("Error in ROC AUC calculation:", e)
                            train_auroc = float(0.5)
                            
                        try:
                            if len(np.unique(train_y)) == 1:
                                print("Only one class present in train_y. Skipping AUPRC calculation.")
                                train_auprc = float(0.5)
                            else:
                                try:
                                    train_auprc = average_precision_score(one_hot(train_y), train_probs)
                                except:
                                    train_auroc = float(0.5)  # 기본값 설정
                        except ValueError as e:
                            print("Error in AUPRC calculation:", e)
                            train_auprc = float(0.5)
                        
                        # train_auroc = roc_auc_score(one_hot(train_y), train_probs)
                        # train_auprc = average_precision_score(one_hot(train_y), train_probs)

                    if wandb:
                        wandb.log({"train_loss": loss.item(), "train_auprc": train_auprc, "train_auroc": train_auroc})
                    if epoch == 0 or epoch == num_epochs - 1:
                        print(confusion_matrix(train_y, np.argmax(train_probs, axis=1), labels=[0, 1]))
                        print('Time: ', end_epoch - start_epoch)

                    """Validation"""
                    model.eval()
                    if epoch == 0 or epoch % 1 == 0:
                        with torch.no_grad():
                            out_val = evaluate_standard(accelerator, model, Pval_tensor, Pval_time_tensor, Pval_static_tensor, valid_paddimg_mask, batch_size=args.batch_size, n_classes=args.n_classes, static=args.static_info)
                            out_val = torch.squeeze(torch.sigmoid(out_val))
                            out_val = out_val.detach().cpu().numpy()

                            val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

                            if dataset == 'P12' or dataset == 'P19':
                                auc_val = roc_auc_score(yval, out_val[:, 1])
                                aupr_val = average_precision_score(yval, out_val[:, 1])
                            elif dataset == 'PAM':
                                auc_val = roc_auc_score(one_hot(yval), out_val)
                                aupr_val = average_precision_score(one_hot(yval), out_val)

                            print("Validation: Epoch %d,  val_loss:%.4f, aupr_val: %.2f, auc_val: %.2f" % (epoch,
                                                                                                            val_loss.item(),
                                                                                                            aupr_val * 100,
                                                                                                            auc_val * 100))
                            if wandb:
                                wandb.log({"val_loss": val_loss.item(), "val_auprc": aupr_val, "val_auroc": auc_val})
                                
                            accelerator.wait_for_everyone()
                            scheduler.step(aupr_val)
                            if auc_val > best_auc_val:
                                
                                best_auc_val = auc_val
                                print(
                                    "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (
                                    epoch, aupr_val * 100, auc_val * 100))
                                patience = 0
                                torch.save(model.state_dict(), model_path + arch + '_' + str(split_idx) + '.pt')
                            else:
                                patience += 1
                                if patience >= 3:
                                    print('early stopping triggered')
                                    break
                                
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                end = time.time()
                time_elapsed = end - start
                print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

                """testing"""
                print('Start Testing')
                model.load_state_dict(torch.load(model_path + arch + '_' + str(split_idx) + '.pt'))
                model = accelerator.prepare(model)
                print('Inference ready')
                model.eval()
                with torch.no_grad():
                    out_test = evaluate(accelerator, model, Ptest_tensor, Ptest_time_tensor, Ptest_static_tensor, test_paddimg_mask, batch_size=args.batch_size, n_classes=args.n_classes, static=args.static_info).numpy()
                        
                    ypred = np.argmax(out_test, axis=1)
                  
                    out_test_max = np.max(out_test, axis=1, keepdims=True)
                    out_test_stable = out_test - out_test_max 

                    denoms = np.sum(np.exp(out_test_stable), axis=1).reshape((-1, 1))
                    probs = np.exp(out_test_stable) / (denoms + 0.00001)


                    acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]

                    if dataset == 'P12' or dataset == 'P19':
                        auc = roc_auc_score(ytest, probs[:, 1])
                        aupr = average_precision_score(ytest, probs[:, 1])
                    elif dataset == 'PAM':
                        auc = roc_auc_score(one_hot(ytest), probs)
                        aupr = average_precision_score(one_hot(ytest), probs)
                        precision = precision_score(ytest, ypred, average='macro')
                        recall = recall_score(ytest, ypred, average='macro')
                        F1 = f1_score(ytest, ypred, average='macro')
                        print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' % (precision * 100, recall * 100, F1 * 100))

                    print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
                    if wandb:
                        wandb.log({"Test AUROC": auc * 100, "Test AUPRC" : aupr * 100,  "Accuracy" :acc * 100})
                    print('classification report', classification_report(ytest, ypred))
                    print(confusion_matrix(ytest, ypred, labels=list(range(args.n_classes))))
                
                
                # store
                acc_arr[k, m] = acc * 100
                auprc_arr[k, m] = aupr * 100
                auroc_arr[k, m] = auc * 100
                if dataset == 'PAM':
                    precision_arr[k, m] = precision * 100
                    recall_arr[k, m] = recall * 100
                    F1_arr[k, m] = F1 * 100
                if wandb:
                    wandb.finish()
        # pick best performer for each split based on max AUPRC
        idx_max = np.argmax(auprc_arr, axis=1)
        acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
        auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
        auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
        if dataset == 'PAM':
            precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
            recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
            F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]

        print("missing ratio:{}, split type:{}, reverse:{}, using baseline:{}".format(missing_ratio, split, reverse,
                                                                                    baseline))
        print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level',
            args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level)

        # display mean and standard deviation
        mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
        mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
        mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
        print('------------------------------------------')
        print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
        print('AUPRC    = %.1f +/- %.1f' % (mean_auprc, std_auprc))
        print('AUROC    = %.1f +/- %.1f' % (mean_auroc, std_auroc))
        if dataset == 'PAM':
            mean_precision, std_precision = np.mean(precision_vec), np.std(precision_vec)
            mean_recall, std_recall = np.mean(recall_vec), np.std(recall_vec)
            mean_F1, std_F1 = np.mean(F1_vec), np.std(F1_vec)
            print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
            print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
            print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))

        # Mark the run as finished
        

        # # save in numpy file
        # np.save('./results/' + arch + '_phy12_setfunction.npy', [acc_vec, auprc_vec, auroc_vec])