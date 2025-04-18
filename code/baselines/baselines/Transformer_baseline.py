import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

wandb = False

if wandb:
    import wandb

    # wandb.offline
    os.environ['WANDB_SILENT']="true"
    wandb.login(key=str('14734fe9c5574e019e8f517149a20d6fe1b2fd0d'))
    config = wandb.config
    # run = wandb.init(project='WBtest', config={'wandb_nb':'wandb_three_in_one_hm'})
    run = wandb.init(project='Raindrop', entity='XZ', config={'wandb_nb':'wandb_three_in_one_hm'})

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from models import TransformerModel, TransformerModel2
from utils_phy12 import *

torch.manual_seed(1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='P19', choices=['P12', 'P19', 'eICU', 'PAM'])
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0')
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--reverse', default=False, help='if True, use female, older for training; if False, use female or younger for training')
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')
parser.add_argument('--imputation', type=str, default='mean', choices=['no_imputation', 'mean', 'forward', 'cubic_spline'],
                    help='use this if you want to impute missing values')

args, unknown = parser.parse_known_args()


arch = 'standard'

model_path = '../../models/'

dataset = args.dataset
print('Dataset used: ', dataset)

if dataset == 'P12':
    base_path = '../../P12data'
elif dataset == 'P19':
    base_path = '../../P19data'
elif dataset == 'eICU':
    base_path = '../../eICUdata'
elif dataset == 'PAM':
    base_path = '../../PAMdata'


def one_hot(y_):
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

feature_removal_level = args.feature_removal_level   # possible values: 'sample', 'set'

if args.withmissingratio == True:
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    missing_ratios = [0]

for missing_ratio in missing_ratios:
    num_epochs = 20
    learning_rate = 0.001

    if dataset == 'P12':
        d_static = 9
        d_inp = 36
        static_info = 1
    elif dataset == 'P19':
        d_static = 6
        d_inp = 34
        static_info = 1
    elif dataset == 'eICU':
        d_static = 399
        d_inp = 14
        static_info = 1
    elif dataset == 'PAM':
        d_static = 0
        d_inp = 17
        static_info = None

    d_model = 36
    nhid = 2 * d_model
    nlayers = 1
    nhead = 1

    dropout = 0.3

    if dataset == 'P12':
        max_len = 215
        n_classes = 2
    elif dataset == 'P19':
        max_len = 60
        n_classes = 2
    elif dataset == 'eICU':
        max_len = 300
        n_classes = 2
    elif dataset == 'PAM':
        max_len = 600
        n_classes = 8

    aggreg = 'mean'

    MAX = 100

    n_runs = 1
    n_splits = 5
    subset = False

    split = args.splittype
    reverse = args.reverse
    baseline = True

    acc_arr = np.zeros((n_splits, n_runs))
    auprc_arr = np.zeros((n_splits, n_runs))
    auroc_arr = np.zeros((n_splits, n_runs))
    precision_arr = np.zeros((n_splits, n_runs))
    recall_arr = np.zeros((n_splits, n_runs))
    F1_arr = np.zeros((n_splits, n_runs))
    for k in range(n_splits):
        split_idx = k + 1
        print('Split id: %d' % split_idx)

        if dataset == 'P12':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'eICU':
            split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'

        # prepare the data:
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split,
                                                                  reverse=reverse, baseline=baseline, dataset=dataset,
                                                                  predictive_label=args.predictive_label)
        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        # impute missing values
        if args.imputation != 'no_imputation':
            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                X_features_train = np.array([d['arr'] for d in Ptrain])
                X_time_train = np.array([d['time'] for d in Ptrain])
                X_features_val = np.array([d['arr'] for d in Pval])
                X_time_val = np.array([d['time'] for d in Pval])
                X_features_test = np.array([d['arr'] for d in Ptest])
                X_time_test = np.array([d['time'] for d in Ptest])
            elif dataset == 'PAM':
                X_features_train = Ptrain
                X_time_train = np.array([np.arange(1, Ptrain.shape[1] + 1)[..., np.newaxis] for d in Ptrain])
                X_features_val = Pval
                X_time_val = np.array([np.arange(1, Pval.shape[1] + 1)[..., np.newaxis] for d in Pval])
                X_features_test = Ptest
                X_time_test = np.array([np.arange(1, Ptest.shape[1] + 1)[..., np.newaxis] for d in Ptest])

            if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM':
                missing_value_num = 0
            elif dataset == 'eICU':
                missing_value_num = -1

            if args.imputation == 'mean':
                features_means = get_features_mean(X_features_train)
                X_features_train = mean_imputation(X_features_train, X_time_train, features_means, missing_value_num)
                X_features_val = mean_imputation(X_features_val, X_time_val, features_means, missing_value_num)
                X_features_test = mean_imputation(X_features_test, X_time_test, features_means, missing_value_num)
            elif args.imputation == 'forward':
                X_features_train = forward_imputation(X_features_train, X_time_train, missing_value_num)
                X_features_val = forward_imputation(X_features_val, X_time_val, missing_value_num)
                X_features_test = forward_imputation(X_features_test, X_time_test, missing_value_num)
            elif args.imputation == 'cubic_spline':
                X_features_train = cubic_spline_imputation(X_features_train, X_time_train, missing_value_num)
                X_features_val = cubic_spline_imputation(X_features_val, X_time_val, missing_value_num)
                X_features_test = cubic_spline_imputation(X_features_test, X_time_test, missing_value_num)

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                for i, pat in enumerate(X_features_train):
                    Ptrain[i]['arr'] = pat
                for i, pat in enumerate(X_features_val):
                    Pval[i]['arr'] = pat
                for i, pat in enumerate(X_features_test):
                    Ptest[i]['arr'] = pat
            elif dataset == 'PAM':
                for i, pat in enumerate(X_features_train):
                    Ptrain[i] = pat
                for i, pat in enumerate(X_features_val):
                    Pval[i] = pat
                for i, pat in enumerate(X_features_test):
                    Ptest[i] = pat

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])

            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            for i in range(len(Ptrain)):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain,
                                                                                                         mf,
                                                                                                         stdf, ms, ss)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf,
                                                                                                 ms, ss)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf,
                                                                                                     stdf, ms,
                                                                                                     ss)

            print(Ptrain_tensor.shape, Ptrain_static_tensor.shape, Ptrain_time_tensor.shape, ytrain_tensor.shape)
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1

            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

        # remove part of variables in validation and test set
        if missing_ratio > 0:
            num_all_features = Pval_tensor.shape[2]
            num_missing_features = round(missing_ratio * num_all_features)
            if feature_removal_level == 'sample':
                for i, patient in enumerate(Pval_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # values
                    Pval_tensor[i] = patient
                for i, patient in enumerate(Ptest_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)   # values
                    Ptest_tensor[i] = patient
            elif feature_removal_level == 'set':
                density_score_indices = np.load('saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
                idx = density_score_indices[:num_missing_features].astype(int)
                Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)
                Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)

        Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
        Pval_tensor = Pval_tensor.permute(1, 0, 2)
        Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

        Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
        Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

        for m in range(n_runs):
            print('- - Run %d - -' % (m + 1))

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                model = TransformerModel2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                          d_static, MAX, 0.5, aggreg, n_classes)
            elif dataset == 'PAM':
                model = TransformerModel2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                          d_static, MAX, 0.5, aggreg, n_classes, static=False)

            model = model.cuda()

            criterion = torch.nn.CrossEntropyLoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                   patience=1, threshold=0.0001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

            # few shot
            print('few shot setting...')
            # 전체 샘플 수
            num_samples = Ptrain_tensor.shape[1]

            # 10% few-shot 샘플 수
            few_shot_ratio = 0.1
            few_shot_size = int(num_samples * few_shot_ratio)

            # 클래스 라벨 추출
            y_all = ytrain_tensor.cpu().numpy()  # 이미 줄이지 않은 전체 ytrain_tensor 기준

            # 클래스별 인덱스
            idx_0 = np.where(y_all == 0)[0]
            idx_1 = np.where(y_all == 1)[0]

            # 클래스 비율 계산
            ratio_0 = len(idx_0) / num_samples
            ratio_1 = len(idx_1) / num_samples

            # 클래스별 추출 샘플 수
            n_0 = int(few_shot_size * ratio_0)
            n_1 = few_shot_size - n_0  # 잔여는 다른 클래스에 할당
            
            # 클래스별로 랜덤 샘플링
            sampled_idx_0 = np.random.choice(idx_0, n_0, replace=False)
            sampled_idx_1 = np.random.choice(idx_1, n_1, replace=False)

            # 최종 few-shot 인덱스
            few_shot_indices = np.concatenate([sampled_idx_0, sampled_idx_1])
            np.random.shuffle(few_shot_indices)  # (선택사항) 셔플

            # 데이터 추출
            Ptrain_tensor = Ptrain_tensor[:, few_shot_indices]
            Ptrain_time_tensor = Ptrain_time_tensor[:, few_shot_indices]
            Ptrain_static_tensor = Ptrain_static_tensor[few_shot_indices]
            ytrain_tensor = ytrain_tensor[few_shot_indices]
            
            ytrain = ytrain_tensor.cpu().numpy() 
            
            
            
            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                strategy = 2
            elif dataset == 'PAM':
                strategy = 3

            n0, n1 = len(idx_0), len(idx_1)
            expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
            expanded_n1 = len(expanded_idx_1)

            batch_size = 128
            if strategy == 1:
                n_batches = 10
            elif strategy == 2:
                K0 = n0 // int(batch_size / 2)
                K1 = expanded_n1 // int(batch_size / 2)
                n_batches = np.min([K0, K1])
            elif strategy == 3:
                n_batches = 30

            best_aupr_val = best_auc_val = 0.0

            start = time.time()
            if wandb:
                wandb.watch(model)
            for epoch in range(num_epochs):
                model.train()

                if strategy == 2:
                    np.random.shuffle(expanded_idx_1)
                    I1 = expanded_idx_1
                    np.random.shuffle(idx_0)
                    I0 = idx_0

                for n in range(n_batches):
                    if strategy == 1:
                        idx = random_sample(idx_0, idx_1, batch_size)
                    elif strategy == 2:
                        """In each batch=128, 64 positive samples, 64 negative samples"""
                        idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                    elif strategy == 3:
                        idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                        # idx = random_sample_8(ytrain, batch_size)   # to balance dataset
                    
                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                    elif dataset == 'PAM':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               None, ytrain_tensor[idx].cuda()

                    lengths = torch.sum(Ptime > 0, dim=0)

                    outputs = evaluate_standard(model, P, Ptime, Pstatic, static=static_info)

                    optimizer.zero_grad()
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    train_probs = torch.squeeze(torch.sigmoid(outputs))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()
                    train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                    train_auprc = average_precision_score(train_y, train_probs[:, 1])
                elif dataset == 'PAM':
                    train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()
                    train_auroc = roc_auc_score(one_hot(train_y), train_probs)
                    train_auprc = average_precision_score(one_hot(train_y), train_probs)

                if wandb:
                    wandb.log({ "train_loss": loss.item(), "train_auprc": train_auprc, "train_auroc": train_auroc})

                """Validation"""
                model.eval()
                if epoch ==0 or epoch % 1 == 0:
                    with torch.no_grad():
                        out_val = evaluate_standard(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor, static=static_info)
                        out_val = torch.squeeze(torch.sigmoid(out_val))
                        out_val = out_val.detach().cpu().numpy()

                        val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

                        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                            auc_val = roc_auc_score(yval, out_val[:, 1])
                            aupr_val = average_precision_score(yval, out_val[:, 1])
                        elif dataset == 'PAM':
                            auc_val = roc_auc_score(one_hot(yval), out_val)
                            aupr_val = average_precision_score(one_hot(yval), out_val)

                        print("Validation: Epoch %d,  val_loss:%.4f, aupr_val: %.2f, auc_val: %.2f" % (epoch,
                          val_loss.item(), aupr_val * 100, auc_val * 100))

                        if wandb:
                            wandb.log({ "val_loss": val_loss.item(), "val_auprc": aupr_val, "val_auroc": auc_val})

                        scheduler.step(aupr_val)
                        if auc_val > best_auc_val:
                            best_auc_val = auc_val
                            print(
                                "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (epoch, aupr_val * 100, auc_val * 100))
                            torch.save(model.state_dict(), model_path + arch + '_' + str(split_idx) + '.pt')

            end = time.time()
            time_elapsed = end - start
            print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

            """Testing"""
            model.load_state_dict(torch.load(model_path + arch + '_' + str(split_idx) + '.pt'))
            model.eval()

            with torch.no_grad():
                out_test = evaluate(model, Ptest_tensor, Ptest_time_tensor, Ptest_static_tensor, n_classes=n_classes, static=static_info).numpy()
                ypred = np.argmax(out_test, axis=1)

                denoms = np.sum(np.exp(out_test), axis=1).reshape((-1, 1))
                probs = np.exp(out_test) / denoms

                acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    auc = roc_auc_score(ytest, probs[:, 1])
                    aupr = average_precision_score(ytest, probs[:, 1])
                elif dataset == 'PAM':
                    auc = roc_auc_score(one_hot(ytest), probs)
                    aupr = average_precision_score(one_hot(ytest), probs)
                    precision = precision_score(ytest, ypred, average='macro', )
                    recall = recall_score(ytest, ypred, average='macro', )
                    F1 = f1_score(ytest, ypred, average='macro', )
                    print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' % (precision * 100, recall * 100, F1 * 100))

                print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
                print('classification report', classification_report(ytest, ypred))
                print(confusion_matrix(ytest, ypred, labels=list(range(n_classes))))

            # store
            acc_arr[k, m] = acc * 100
            auprc_arr[k, m] = aupr * 100
            auroc_arr[k, m] = auc * 100
            if dataset == 'PAM':
                precision_arr[k, m] = precision * 100
                recall_arr[k, m] = recall * 100
                F1_arr[k, m] = F1 * 100

    # pick best performer for each split based on max AUPRC
    idx_max = np.argmax(auprc_arr, axis=1)
    acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
    auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
    auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
    if dataset == 'PAM':
        precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
        recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
        F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]

    print("missing ratio:{}, split type:{}, reverse:{}, using baseline:{}".format(missing_ratio, split, reverse, baseline))

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

    if wandb:
        wandb.finish()

    # # save in numpy file
    # np.save('./results/' + arch + '_phy12_setfunction.npy', [acc_vec, auprc_vec, auroc_vec])
