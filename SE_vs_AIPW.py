import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from model_blocks import MLP1

np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def compute_harm(datasets, rate,  trials):
    if datasets == 'ihdp':
        train_data = np.load('data/ihdp_npci_1-100.train.npz')

        x_tr = train_data['x'][:, :, 0]
        t_tr = train_data['t'][:, 0]
        x_tr = (x_tr - np.mean(x_tr, axis=0)) / np.std(x_tr, axis=0)

        # This is how we simulate the potential outcome. For IHDP, alpha0 is 1 and alpha 1 is 3.
        w0 = np.clip(np.random.normal(0, 1, x_tr.shape[1]), -1, 1)
        w1 = 2 * np.random.random(x_tr.shape[1]) - 1
        y0_tr = np.random.binomial(1, sigmoid(np.sum(w0 * x_tr, axis=1) + np.random.normal(1, 1, x_tr.shape[0])))
        y1_tr = np.random.binomial(1, sigmoid(np.sum(w1 * x_tr, axis=1) + np.random.normal(3, 1, x_tr.shape[0])))
        y_tr = np.where(t_tr == 0, y0_tr, y1_tr)

        harm = []
        for i in range(len(y0_tr)):
            if y0_tr[i] == 1 and y1_tr[i] == 0:
                harm.append(1)
            else:
                harm.append(0)
        harm = np.array(harm)

    if datasets == 'jobs':
        train_data = np.load('data/jobs_DW_bin.new.10.train.npz')

        x_tr = train_data['x'][:, :, 0]
        t_tr = train_data['t'][:, 0]
        x_tr = (x_tr - np.mean(x_tr, axis=0)) / np.std(x_tr, axis=0)

        # This is how we simulate the potential outcome. For JOBS, alpha0 is 0 and alpha1 is 2
        w0 = np.clip(np.random.normal(0, 1, x_tr.shape[1]), -1, 1)
        w1 = 2 * np.random.random(x_tr.shape[1]) - 1
        y0_tr = np.random.binomial(1, sigmoid(np.sum(w0 * x_tr, axis=1) + np.random.normal(0, 1, x_tr.shape[0])))
        y1_tr = np.random.binomial(1,sigmoid(np.sum(w1 * x_tr, axis=1) + np.random.normal(2, 1, x_tr.shape[0])))
        y_tr = np.where(t_tr == 0, y0_tr, y1_tr)

        harm = []
        for i in range(len(y0_tr)):
            if y0_tr[i] == 1 and y1_tr[i] == 0:
                harm.append(1)
            else:
                harm.append(0)
        harm = np.array(harm)

    pred_1 = np.ones((int(len(x_tr) * rate), 1))
    pred_2 = np.zeros((len(x_tr)-int(len(x_tr) * rate), 1))
    pred = np.vstack((pred_1, pred_2))
    pred = np.squeeze(pred, axis=1)
    true_harm = np.sum(pred * harm)

    result = np.zeros((trials, 3))
    bias = np.zeros((trials, 2))
    for i in range(trials):

        mlp_p = MLP1(input_size=x_tr.shape[1])
        mlp_p.fit(x_tr, t_tr, num_epoch=1000, lr=0.01, batch_size=128)
        p_tr = mlp_p.predict(x_tr)
        p_tr = np.clip(p_tr, 0.1, 0.9)


        clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
        clf.fit(x_tr[t_tr == 0], y_tr[t_tr == 0])
        y0_or = clf.predict_proba(x_tr)[:, 1]
        clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
        clf.fit(x_tr[t_tr == 1], y_tr[t_tr == 1])
        y1_or = clf.predict_proba(x_tr)[:, 1]

        # Consturcted mu0 and mu1 in AIPW
        mlp = MLP1(input_size=x_tr.shape[1])
        mlp.fit(x_tr[t_tr == 0],
                (y_tr[t_tr == 0] / (1 - p_tr[t_tr == 0]) - y0_or[t_tr == 0] / (1 - p_tr[t_tr == 0])) + y0_or[t_tr == 0])
        y0_dr = mlp.predict(x_tr)
        mlp = MLP1(input_size=x_tr.shape[1])
        mlp.fit(x_tr[t_tr == 1],
                (y_tr[t_tr == 1] / (p_tr[t_tr == 1]) - y1_or[t_tr == 1] / (p_tr[t_tr == 1])) + y1_or[t_tr == 1],
                batch_size=20)
        y1_dr = mlp.predict(x_tr)

        AIPW_harm = np.sum(((1 - t_tr) * (y_tr - y0_or) / (1 - p_tr) + y0_or) * pred -
                                      (t_tr * (y_tr - y0_or) / p_tr + y1_or) * y0_or * pred)

        SE_harm = np.sum(((1 - t_tr) * (y_tr - y0_or) / (1 - p_tr) + y0_or) * (1 - y1_or) * pred -(t_tr * (y_tr - y1_or) / p_tr) * y0_or * pred)

        result[i, 0] = true_harm
        result[i, 1] = AIPW_harm
        result[i, 2] = SE_harm
        bias[i, 0] = np.abs(AIPW_harm - true_harm)
        bias[i, 1] = np.abs(SE_harm - true_harm)

    resultmean = result.mean(axis=0)
    biasmean = bias.mean(axis=0)
    resultstd = result[:,1:].std(axis=0)

    return resultmean, biasmean, resultstd


if __name__ == '__main__':
    datasets = ['ihdp','jobs']
    rates = [0.2,0.5]
    trials = 10
    for dataset in datasets:
        for rate in rates:
            resultmean, biasmean, resultstd = compute_harm(dataset, rate, trials)
            print("dataset=",dataset)
            print("rate=",rate)
            print(resultmean,biasmean,resultstd)
