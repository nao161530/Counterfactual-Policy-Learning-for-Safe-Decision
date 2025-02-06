import numpy as np
from model_blocks_aipw import model_cate_aipw, model_lagrange_aipw, model_constraint_aipw, model_linear_aipw
from model_blocks_se import model_cate_se, model_lagrange_se, model_constraint_se, model_linear_se


def se(datasets,x_tr, t_tr, y_tr, p_tr, y0_tr, y1_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm, harm_rate, npref, pref_idx,num_epoch, lr, batch_size, panelty, trials):
    harm_num = sum(harm)
    result = np.zeros((trials, 21))
    for i in range(trials):
        print("datasets={},pref_idx={},trials={}:".format(datasets, pref_idx, i))

        OR = model_cate_se(input_size=x_tr.shape[1])
        OR.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho,num_epoch=num_epoch, lr=lr, batch_size=batch_size, lamb=1e-1, tol=6)

        OR_lagrange = model_lagrange_se(input_size=x_tr.shape[1])
        OR_lagrange.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm_num=int(harm_num * 0.2), thr=-0.3 * int(harm_num * 0.2),
                          stop=10, lr=lr, num_epoch=num_epoch, batch_size=batch_size, panelty=panelty, tol=1e-4, lamb=1e-1)

        OR_constraint = model_constraint_se(input_size=x_tr.shape[1])
        total_epsilon = OR_constraint.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm_rate, npref, pref_idx,lr=lr, num_epoch=num_epoch, batch_size=batch_size, panelty=panelty, tol=1e-4, lamb=1e-1)

        OR_linear = model_linear_se(input_size=x_tr.shape[1])
        OR_linear.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, npref, pref_idx,lr=lr, num_epoch=num_epoch, batch_size=batch_size, stop=10, tol=1e-4, lamb=1e-1)

        pred = OR.predict(x_tr)
        pred_lagrange = OR_lagrange.predict(x_tr)
        pred_constraint = OR_constraint.predict(x_tr)
        pred_linear = OR_linear.predict(x_tr)

        y_tr_cate = np.where(pred == 0, y0_tr, y1_tr)
        y_tr_lagrange = np.where(pred_lagrange == 0, y0_tr, y1_tr)
        y_tr_constraint = np.where(pred_constraint == 0, y0_tr, y1_tr)
        y_tr_linear = np.where(pred_linear == 0, y0_tr, y1_tr)

        # reward
        result[i, 0] = sum(y_tr_cate - pred * c)
        result[i, 1] = sum(y_tr_lagrange - pred_lagrange * c)
        result[i, 2] = sum(y_tr_constraint - pred_constraint * c)
        result[i, 3] = sum(y_tr_linear - pred_linear * c)

        # welfare
        welfare = y1_tr - y0_tr
        result[i, 4] = sum(welfare[pred == 1])
        result[i, 5] = sum(welfare[pred_lagrange == 1])
        result[i, 6] = sum(welfare[pred_constraint == 1])
        result[i, 7] = sum(welfare[pred_linear == 1])

        # harm
        result[i, 8] = sum(harm[pred == 1])
        result[i, 9] = sum(harm[pred_lagrange == 1])
        result[i, 10] = sum(harm[pred_constraint == 1])
        result[i, 11] = sum(harm[pred_linear == 1])


        result[i, 12] = total_epsilon

    resultmean = result.mean(axis=0)
    resultstd = result.std(axis=0)

    return resultmean, resultstd

def aipw(datasets,x_tr, t_tr, y_tr, p_tr, y0_tr, y1_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm, harm_rate, npref, pref_idx,num_epoch,lr, batch_size, panelty, trials):
    harm_num = sum(harm)
    result = np.zeros((trials, 21))
    for i in range(trials):
        print("datasets={},pref_idx={},trials={}:".format(datasets, pref_idx, i))

        OR = model_cate_aipw(input_size=x_tr.shape[1])
        OR.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho,num_epoch=num_epoch, lr=lr, batch_size=batch_size, lamb=1e-1, tol=6)

        OR_lagrange = model_lagrange_aipw(input_size=x_tr.shape[1])
        OR_lagrange.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm_num=int(harm_num * 0.2), thr=-0.3 * int(harm_num * 0.2),
                          stop=10, lr=lr, num_epoch=num_epoch, batch_size=batch_size, panelty=panelty, tol=1e-4, lamb=1e-1)

        OR_constraint = model_constraint_aipw(input_size=x_tr.shape[1])
        total_epsilon = OR_constraint.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, harm_rate, npref, pref_idx,lr=lr, num_epoch=num_epoch, batch_size=batch_size, panelty=panelty, tol=1e-4, lamb=1e-1)

        OR_linear = model_linear_aipw(input_size=x_tr.shape[1])
        OR_linear.fit(x_tr, t_tr, y_tr, p_tr, y0_or, y1_or, y0_dr, y1_dr, c, rho, npref, pref_idx,lr=lr, num_epoch=num_epoch, batch_size=batch_size, stop=10, tol=1e-4, lamb=1e-1)

        pred = OR.predict(x_tr)
        pred_lagrange = OR_lagrange.predict(x_tr)
        pred_constraint = OR_constraint.predict(x_tr)
        pred_linear = OR_linear.predict(x_tr)

        y_tr_cate = np.where(pred == 0, y0_tr, y1_tr)
        y_tr_lagrange = np.where(pred_lagrange == 0, y0_tr, y1_tr)
        y_tr_constraint = np.where(pred_constraint == 0, y0_tr, y1_tr)
        y_tr_linear = np.where(pred_linear == 0, y0_tr, y1_tr)

        # reward
        result[i, 0] = sum(y_tr_cate - pred * c)
        result[i, 1] = sum(y_tr_lagrange - pred_lagrange * c)
        result[i, 2] = sum(y_tr_constraint - pred_constraint * c)
        result[i, 3] = sum(y_tr_linear - pred_linear * c)

        # welfare
        welfare = y1_tr - y0_tr
        result[i, 4] = sum(welfare[pred == 1])
        result[i, 5] = sum(welfare[pred_lagrange == 1])
        result[i, 6] = sum(welfare[pred_constraint == 1])
        result[i, 7] = sum(welfare[pred_linear == 1])

        # harm
        result[i, 8] = sum(harm[pred == 1])
        result[i, 9] = sum(harm[pred_lagrange == 1])
        result[i, 10] = sum(harm[pred_constraint == 1])
        result[i, 11] = sum(harm[pred_linear == 1])


        result[i, 12] = total_epsilon

    resultmean = result.mean(axis=0)
    resultstd = result.std(axis=0)

    return resultmean, resultstd

