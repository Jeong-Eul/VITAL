
import numpy as np
import torch


def random_split(n=11988, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Use 9:1:1 split"""
    p_train = train_ratio
    p_val = val_ratio
    p_test = test_ratio

    n = 11988  # original 12000 patients, remove 12 outliers
    n_train = round(n * p_train)
    n_val = round(n * p_val)
    n_test = n - (n_train + n_val)
    p = np.random.permutation(n)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train + n_val]
    idx_test = p[n_train + n_val:]
    return idx_train, idx_val, idx_test


def get_data_split(base_path, split_path, split_type='random', reverse=False, baseline=True, dataset='P12', predictive_label='mortality'):
    # load data
    if dataset == 'P12':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'eICU':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = 'eICU_'
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

    show_statistics = False
    if show_statistics:
        idx_under_65 = []
        idx_over_65 = []

        idx_male = []
        idx_female = []

        # variables for statistics
        all_ages = []
        female_count = 0
        male_count = 0
        all_BMI = []

        X_static = np.zeros((len(Pdict_list), len(Pdict_list[0]['extended_static'])))
        for i in range(len(Pdict_list)):
            X_static[i] = Pdict_list[i]['extended_static']
            age, gender_0, gender_1, height, _, _, _, _, weight = X_static[i]
            if age > 0:
                all_ages.append(age)
                if age < 65:
                    idx_under_65.append(i)
                else:
                    idx_over_65.append(i)
            if gender_0 == 1:
                female_count += 1
                idx_female.append(i)
            if gender_1 == 1:
                male_count += 1
                idx_male.append(i)
            if height > 0 and weight > 0:
                all_BMI.append(weight / ((height / 100) ** 2))

        # # plot statistics
        # plt.hist(all_ages, bins=[i * 10 for i in range(12)])
        # plt.xlabel('Years')
        # plt.ylabel('# people')
        # plt.title('Histogram of patients ages, age known in %d samples.\nMean: %.1f, Std: %.1f, Median: %.1f' %
        #           (len(all_ages), np.mean(np.array(all_ages)), np.std(np.array(all_ages)), np.median(np.array(all_ages))))
        # plt.show()
        #
        # plt.hist(all_BMI, bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        # all_BMI = np.array(all_BMI)
        # all_BMI = all_BMI[(all_BMI > 10) & (all_BMI < 65)]
        # plt.xlabel('BMI')
        # plt.ylabel('# people')
        # plt.title('Histogram of patients BMI, height and weight known in %d samples.\nMean: %.1f, Std: %.1f, Median: %.1f' %
        #           (len(all_BMI), np.mean(all_BMI), np.std(all_BMI), np.median(all_BMI)))
        # plt.show()
        # print('\nGender known: %d,  Male count: %d,  Female count: %d\n' % (male_count + female_count, male_count, female_count))

    # np.save('saved/idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
    # np.save('saved/idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
    # np.save('saved/idx_male.npy', np.array(idx_male), allow_pickle=True)
    # np.save('saved/idx_female.npy', np.array(idx_female), allow_pickle=True)

    if baseline==True:
        BL_path = ''
    else:
        BL_path = 'baselines/'

    if split_type == 'random':
        # load random indices from a split
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
    elif split_type == 'age':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]
    elif split_type == 'gender':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM':
        if predictive_label == 'mortality':
            y = arr_outcomes[:, -1].reshape((-1, 1))
        elif predictive_label == 'LoS':  # for P12 only
            y = arr_outcomes[:, 3].reshape((-1, 1))
            y = np.array(list(map(lambda los: 0 if los <= 3 else 1, y)))[..., np.newaxis]
    elif dataset == 'eICU':
        y = arr_outcomes[..., np.newaxis]
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.maximum(stdf[f], eps)
    return mf, stdf


def mask_normalize(P_tensor, mf, stdf, lengths):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = 1*(P_tensor > 0) + 0*(P_tensor <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    M_null = np.zeros((N, T, F))
    for i in range(N):
        length = lengths[i]  

        M_null[i, :length, :] = 1 
        M_null[i, :length, :][P_tensor[i, :length, :] == 0] = np.nan
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor, M_null


def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == 'eICU':
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical[s] == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss


def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor


def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_static_tensor = np.zeros((len(P), D))
    lengths = np.zeros(len(P), dtype=int)
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_static_tensor[i] = P[i]['extended_static']
        lengths[i] = P[i]['length']
    P_tensor, M_null = mask_normalize(P_tensor, mf, stdf, lengths)
    P_tensor = torch.Tensor(P_tensor)
    M_null = torch.Tensor(M_null)
    
    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, P_static_tensor, P_time, y_tensor, M_null

def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    lengths = np.zeros(len(P), dtype=int)
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
        lengths[i] = T
    P_tensor, M_null = mask_normalize(P, mf, stdf, lengths)
    M_null = torch.Tensor(M_null)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor, M_null


def masked_softmax(A, epsilon=0.000000001):
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * (A != 0).float()
    A_softmax = A_exp / (torch.sum(A_exp, dim=0, keepdim=True) + epsilon)
    return A_softmax


def random_sample(idx_0, idx_1, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(idx_0, size=int(B / 2), replace=replace)
    idx1_batch = np.random.choice(idx_1, size=int(B / 2), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


def evaluate(accelerator, model, P_tensor, P_time_tensor, P_static_tensor, mask, batch_size=128, n_classes=2, static=1):
    model.eval()
    P_tensor = P_tensor[:, :, :int(P_tensor.shape[2] / 2)].to(accelerator.device)
    P_time_tensor = P_time_tensor.to(accelerator.device) 
    mask = mask.to(accelerator.device)
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.to(accelerator.device)
        N, Fs = P_static_tensor.shape

    N, T, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[start:start + batch_size, :, :]
        M = mask[start:start + batch_size, :, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        real_time = torch.sum(Ptime > 0, dim=0)
        middleoutput = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem, :, :]
        M = mask[start:start + rem, :, :]
        Ptime = P_time_tensor[:, start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        # 데이터가 유효한지 확인
        if P.shape[0] == 0 or M.shape[0] == 0:
            print('dd')
            return out  # 빈 배치가 있는 경우 이미 처리한 결과 반환    
        
        real_time = torch.sum(Ptime > 0, dim=0)
        whatever = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        out[start:start + rem] = whatever.detach().cpu()
    return out


def evaluate_standard(accelerator, model, P_tensor, P_time_tensor, P_static_tensor, mask, batch_size=128, n_classes=2, static=1):
    model.eval()
    P_tensor = P_tensor[:, :, :int(P_tensor.shape[2] / 2)].to(accelerator.device)
    P_time_tensor = P_time_tensor.to(accelerator.device) 
    mask = mask.to(accelerator.device)
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.to(accelerator.device)
        N, Fs = P_static_tensor.shape

    N, T, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[start:start + batch_size, :, :]
        M = mask[start:start + batch_size, :, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        real_time = torch.sum(Ptime > 0, dim=0)
        middleoutput = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem, :, :]
        M = mask[start:start + rem, :, :]
        Ptime = P_time_tensor[:, start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        real_time = torch.sum(Ptime > 0, dim=0)
        whatever = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        out[start:start + rem] = whatever.detach().cpu()
    return out


def evaluate_v2(accelerator, model, P_tensor, P_time_tensor, P_static_tensor, mask, length, batch_size=30, n_classes=2, static=1):
    model.eval()
    P_tensor = P_tensor[:, :, :int(P_tensor.shape[2] / 2)].to(accelerator.device)
    P_time_tensor = P_time_tensor.to(accelerator.device) 
    mask = mask.to(accelerator.device)
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.to(accelerator.device)
        N, Fs = P_static_tensor.shape

    N, T, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    labout = torch.zeros(N, length, 32)
    start = 0
    for i in range(n_batches):
        P = P_tensor[start:start + batch_size, :, :]
        M = mask[start:start + batch_size, :, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        real_time = torch.sum(Ptime > 0, dim=0)
        _, laboutput = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        labout[start:start + batch_size] = laboutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem, :, :]
        M = mask[start:start + rem, :, :]
        Ptime = P_time_tensor[:, start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        # 데이터가 유효한지 확인
        if P.shape[0] == 0 or M.shape[0] == 0:
            print('dd')
            return labout  # 빈 배치가 있는 경우 이미 처리한 결과 반환    
        
        real_time = torch.sum(Ptime > 0, dim=0)
        _, whatever = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        labout[start:start + rem] = whatever.detach().cpu()
    return labout


def evaluate_standard_v2(accelerator, model, P_tensor, P_time_tensor, P_static_tensor, mask, length, batch_size=30, n_classes=2, static=1):
    model.eval()
    P_tensor = P_tensor[:, :, :int(P_tensor.shape[2] / 2)].to(accelerator.device)
    P_time_tensor = P_time_tensor.to(accelerator.device) 
    mask = mask.to(accelerator.device)
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.to(accelerator.device)
        N, Fs = P_static_tensor.shape

    N, T, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    labout = torch.zeros(N, length, 32)
    start = 0
    for i in range(n_batches):
        P = P_tensor[start:start + batch_size, :, :]
        M = mask[start:start + batch_size, :, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        real_time = torch.sum(Ptime > 0, dim=0)
        _, laboutput = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        labout[start:start + batch_size] = laboutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem, :, :]
        M = mask[start:start + rem, :, :]
        Ptime = P_time_tensor[:, start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        real_time = torch.sum(Ptime > 0, dim=0)
        _, whatever = model.forward(P, Ptime.permute(1, 0).unsqueeze(-1), real_time, Pstatic, M)
        labout[start:start + rem] = whatever.detach().cpu()
    return labout
