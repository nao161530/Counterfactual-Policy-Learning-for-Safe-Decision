import numpy as np
import pickle
from train import se,aipw

def pred_se(trials, datasets, c, npref, pref_idx, num_epoch, lr, batch_size, panelty):
    rho = 1
    if datasets == "ihdp":
        train_data = np.load('data/ihdp_npci_1-100.train.npz')

        x_tr = train_data['x'][:, :, 0]
        t_tr = train_data['t'][:, 0]
        x_tr = (x_tr - np.mean(x_tr, axis=0)) / np.std(x_tr, axis=0)

        with open("data/IHDP/ihdp1_train_data", "rb") as file:
            y0_tr = pickle.load(file)
            y1_tr = pickle.load(file)
            y0_or = pickle.load(file)
            y1_or = pickle.load(file)
            y0_ips = pickle.load(file)
            y1_ips = pickle.load(file)
            y0_dr = pickle.load(file)
            y1_dr = pickle.load(file)
            p_tr = pickle.load(file)

        y_tr = np.where(t_tr == 0, y0_tr, y1_tr)
        harm = []
        for i in range(len(y0_tr)):
            if y0_tr[i] == 1 and y1_tr[i] == 0:
                harm.append(1)
            else:
                harm.append(0)
        harm = np.array(harm)

        harm_rate = y0_or * (1 - y1_or)


    elif datasets == 'jobs':
        train_data = np.load('data/jobs_DW_bin.new.10.train.npz')

        x_tr = train_data['x'][:, :, 0]
        t_tr = train_data['t'][:, 0]
        x_tr = (x_tr - np.mean(x_tr, axis=0)) / np.std(x_tr, axis=0)

        with open("data/JOBS/jobs_train_data", "rb") as file:
            y0_tr = pickle.load(file)
            y1_tr = pickle.load(file)
            y0_or = pickle.load(file)
            y1_or = pickle.load(file)
            y0_ips = pickle.load(file)
            y1_ips = pickle.load(file)
            y0_dr = pickle.load(file)
            y1_dr = pickle.load(file)
            p_tr = pickle.load(file)

        y_tr = np.where(t_tr == 0, y0_tr, y1_tr)


        harm = []
        for i in range(len(y0_tr)):
            if y0_tr[i] == 1 and y1_tr[i] == 0:
                harm.append(1)
            else:
                harm.append(0)
        harm = np.array(harm)

        harm_rate = y0_or * (1 - y1_or)

    result_se, std_se = se(datasets, x_tr, t_tr, y_tr, p_tr, y0_tr, y1_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm, harm_rate, npref, pref_idx, num_epoch, lr, batch_size, panelty, trials)
    list_se = [datasets, c,  pref_idx, result_se[0], result_se[1], result_se[2], result_se[3], result_se[4],
                 result_se[5], result_se[6], result_se[7], result_se[8],
                 result_se[9], result_se[10], result_se[11],result_se[0]/result_se[8],result_se[1]/result_se[9],result_se[2]/result_se[10],result_se[3]/result_se[11], result_se[12],
                 std_se[0], std_se[1], std_se[2],std_se[3], std_se[4], std_se[5]
        , std_se[6], std_se[7], std_se[8], std_se[9], std_se[10], std_se[11],std_se[12], panelty]

    return list_se

def pred_aipw(trials, datasets, c, npref, pref_idx, num_epoch, lr, batch_size, panelty):
    rho = 1
    if datasets == "ihdp":
        train_data = np.load('data/ihdp_npci_1-100.train.npz')

        x_tr = train_data['x'][:, :, 0]
        t_tr = train_data['t'][:, 0]
        x_tr = (x_tr - np.mean(x_tr, axis=0)) / np.std(x_tr, axis=0)


        with open("data/IHDP/ihdp1_train_data", "rb") as file:
            y0_tr = pickle.load(file)
            y1_tr = pickle.load(file)
            y0_or = pickle.load(file)
            y1_or = pickle.load(file)
            y0_ips = pickle.load(file)
            y1_ips = pickle.load(file)
            y0_dr = pickle.load(file)
            y1_dr = pickle.load(file)
            p_tr = pickle.load(file)

        y_tr = np.where(t_tr == 0, y0_tr, y1_tr)
        harm = []
        for i in range(len(y0_tr)):
            if y0_tr[i] == 1 and y1_tr[i] == 0:
                harm.append(1)
            else:
                harm.append(0)
        harm = np.array(harm)

        harm_rate = y0_or * (1 - y1_or)


    elif datasets == 'jobs':
        train_data = np.load('data/jobs_DW_bin.new.10.train.npz')

        x_tr = train_data['x'][:, :, 0]
        t_tr = train_data['t'][:, 0]
        x_tr = (x_tr - np.mean(x_tr, axis=0)) / np.std(x_tr, axis=0)

        with open("data/JOBS/jobs_train_data", "rb") as file:
            y0_tr = pickle.load(file)
            y1_tr = pickle.load(file)
            y0_or = pickle.load(file)
            y1_or = pickle.load(file)
            y0_ips = pickle.load(file)
            y1_ips = pickle.load(file)
            y0_dr = pickle.load(file)
            y1_dr = pickle.load(file)
            p_tr = pickle.load(file)

        y_tr = np.where(t_tr == 0, y0_tr, y1_tr)


        harm = []
        for i in range(len(y0_tr)):
            if y0_tr[i] == 1 and y1_tr[i] == 0:
                harm.append(1)
            else:
                harm.append(0)
        harm = np.array(harm)

        harm_rate = y0_or * (1 - y1_or)

    result_aipw, std_aipw = aipw(datasets, x_tr, t_tr, y_tr, p_tr, y0_tr, y1_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm,harm_rate, npref, pref_idx, num_epoch, lr, batch_size, panelty, trials)
    list_aipw = [datasets, c, pref_idx, result_aipw[0], result_aipw[1], result_aipw[2], result_aipw[3], result_aipw[4],
                 result_aipw[5], result_aipw[6], result_aipw[7], result_aipw[8],
                 result_aipw[9], result_aipw[10], result_aipw[11],result_aipw[0]/result_aipw[8],result_aipw[1]/result_aipw[9],result_aipw[2]/result_aipw[10],result_aipw[3]/result_aipw[11], result_aipw[12],
                 std_aipw[0], std_aipw[1], std_aipw[2],std_aipw[3], std_aipw[4], std_aipw[5]
        , std_aipw[6], std_aipw[7], std_aipw[8], std_aipw[9], std_aipw[10], std_aipw[11],std_aipw[12], panelty]

    return list_aipw


if __name__ == '__main__':
    from joblib import Parallel, delayed
    from pandas.core.frame import DataFrame
    import os

    trials = 20
    npref = 10
    datasets = 'ihdp'
    para = []
    batch_size = 256
    num_epoch = 300
    lr = 0.005
    panelty_list = [225]
    for panelty in panelty_list:
        cost = [0,0.05,0.10]
        for c in cost:
            for pref_idx in range(npref):
                paralist = [trials, datasets, c, npref, pref_idx, num_epoch, lr, batch_size, panelty]
                para.append(paralist)

    result_se = Parallel(n_jobs=-1)(delayed(pred_se)(*arglist) for arglist in para)
    result_aipw = Parallel(n_jobs=-1)(delayed(pred_aipw)(*arglist) for arglist in para)

    data_se = DataFrame(result_se)
    data_aipw = DataFrame(result_aipw)

    data_se.rename(columns={0: 'datasets', 1: 'cost', 2: 'pref_idx', 3: 'reward_cate', 4: 'reward_lagrange',
                              5: 'reward_constraint', 6: 'reward_linear', 7: 'welfare_cate',
                              8: 'welfare_lagrange', 9: 'welfare_constraint',
                              10: 'welfare_linear', 11: 'harm_cate', 12: 'harm_lagrange',
                              13: 'harm_constraint', 14: 'harm_linear', 15: 'rr_cate',
                              16: 'rr_lagrange',17: 'rr_constraint', 18: 'rr_linear', 19: 'epsilon',
                              20: 'std_cate_reward', 21: 'std_lagrange_reward', 22: 'std_constraint_reward',
                              23: 'std_linear_reward', 24: 'std_cate_welfare',
                              25: 'std_lagrange_welfare', 26: 'std_constraint_welfare', 27: 'std_linear_welfare', 28: 'std_cate_harm', 29: 'std_lagrange_harm',
                              30: 'std_constraint_harm', 31: 'std_linear_harm', 32: 'std_epsilon',33:'panelty'},
                     inplace=True)

    data_aipw.rename(columns={0: 'datasets', 1: 'cost', 2: 'pref_idx', 3: 'reward_cate', 4: 'reward_lagrange',
                              5: 'reward_constraint', 6: 'reward_linear', 7: 'welfare_cate',
                              8: 'welfare_lagrange', 9: 'welfare_constraint',
                              10: 'welfare_linear', 11: 'harm_cate', 12: 'harm_lagrange',
                              13: 'harm_constraint', 14: 'harm_linear', 15: 'rr_cate',
                              16: 'rr_lagrange',17: 'rr_constraint', 18: 'rr_linear', 19: 'epsilon',
                              20: 'std_cate_reward', 21: 'std_lagrange_reward', 22: 'std_constraint_reward',
                              23: 'std_linear_reward', 24: 'std_cate_welfare',
                              25: 'std_lagrange_welfare', 26: 'std_constraint_welfare', 27: 'std_linear_welfare', 28: 'std_cate_harm', 29: 'std_lagrange_harm',
                              30: 'std_constraint_harm', 31: 'std_linear_harm', 32: 'std_epsilon',33:'panelty'},
                     inplace=True)


    file_path = "result/" + datasets + "/npref_" + str(npref) + "/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    data_se.to_csv(file_path + "result_se.csv", index=False, float_format='%.4f')
    data_aipw.to_csv(file_path + "result_aipw.csv",index=False,float_format='%.4f')
###########################################################################################

    datasets = 'jobs'
    para = []
    batch_size = 512
    num_epoch = 300
    lr = 0.005
    panelty_list = [20]
    for panelty in panelty_list:
        cost = [0,0.05,0.10]
        for c in cost:
            for pref_idx in range(npref):
                # result_aipw = pred_aipw(trials, dataset, c, npref, pref_idx, num_epoch, lr, batch_size, panelty)
                paralist = [trials, datasets, c, npref, pref_idx, num_epoch, lr, batch_size, panelty]
                para.append(paralist)

    result_se = Parallel(n_jobs=-1)(delayed(pred_se)(*arglist) for arglist in para)
    result_aipw = Parallel(n_jobs=-1)(delayed(pred_aipw)(*arglist) for arglist in para)

    data_se = DataFrame(result_se)
    data_aipw = DataFrame(result_aipw)

    data_se.rename(columns={0: 'datasets', 1: 'cost', 2: 'pref_idx', 3: 'reward_cate', 4: 'reward_lagrange',
                              5: 'reward_constraint', 6: 'reward_linear', 7: 'welfare_cate',
                              8: 'welfare_lagrange', 9: 'welfare_constraint',
                              10: 'welfare_linear', 11: 'harm_cate', 12: 'harm_lagrange',
                              13: 'harm_constraint', 14: 'harm_linear', 15: 'rr_cate',
                              16: 'rr_lagrange',17: 'rr_constraint', 18: 'rr_linear', 19: 'epsilon',
                              20: 'std_cate_reward', 21: 'std_lagrange_reward', 22: 'std_constraint_reward',
                              23: 'std_linear_reward', 24: 'std_cate_welfare',
                              25: 'std_lagrange_welfare', 26: 'std_constraint_welfare', 27: 'std_linear_welfare', 28: 'std_cate_harm', 29: 'std_lagrange_harm',
                              30: 'std_constraint_harm', 31: 'std_linear_harm', 32: 'std_epsilon',33:'panelty'},
                     inplace=True)

    data_aipw.rename(columns={0: 'datasets', 1: 'cost', 2: 'pref_idx', 3: 'reward_cate', 4: 'reward_lagrange',
                              5: 'reward_constraint', 6: 'reward_linear', 7: 'welfare_cate',
                              8: 'welfare_lagrange', 9: 'welfare_constraint',
                              10: 'welfare_linear', 11: 'harm_cate', 12: 'harm_lagrange',
                              13: 'harm_constraint', 14: 'harm_linear', 15: 'rr_cate',
                              16: 'rr_lagrange',17: 'rr_constraint', 18: 'rr_linear', 19: 'epsilon',
                              20: 'std_cate_reward', 21: 'std_lagrange_reward', 22: 'std_constraint_reward',
                              23: 'std_linear_reward', 24: 'std_cate_welfare',
                              25: 'std_lagrange_welfare', 26: 'std_constraint_welfare', 27: 'std_linear_welfare', 28: 'std_cate_harm', 29: 'std_lagrange_harm',
                              30: 'std_constraint_harm', 31: 'std_linear_harm', 32: 'std_epsilon',33:'panelty'},
                     inplace=True)


    file_path = "result/" + datasets + "/npref_" + str(npref) + "/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    data_se.to_csv(file_path + "result_se.csv", index=False, float_format='%.4f')
    data_aipw.to_csv(file_path + "result_aipw.csv", index=False, float_format='%.4f')

print("Done!")