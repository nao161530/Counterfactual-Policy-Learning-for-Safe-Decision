import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from model_blocks import MLP1

train_data = np.load('data/jobs_DW_bin.new.10.train.npz')

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

x_tr = train_data['x'][:, :, 0]
t_tr = train_data['t'][:, 0]
x_tr = (x_tr - np.mean(x_tr, axis = 0))/np.std(x_tr, axis = 0)

# This is how we simulate the potential outcome. For JOBS, alpha0 is 0 and alpha1 is 2
w0 = np.clip(np.random.normal(0, 1, x_tr.shape[1]), -1, 1)
w1 = 2 * np.random.random(x_tr.shape[1]) - 1
y0_tr = np.random.binomial(1, sigmoid(np.sum(w0*x_tr, axis = 1) + np.random.normal(0, 1, x_tr.shape[0])))
y1_tr = np.random.binomial(1, sigmoid(np.sum(w1*x_tr, axis = 1) + np.random.normal(2, 1, x_tr.shape[0])))
y_tr = np.where(t_tr == 0, y0_tr, y1_tr)

# Constructed propensity score
clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
clf.fit(x_tr, t_tr)
p_tr = clf.predict_proba(x_tr)[:, 1]
p_tr = np.clip(p_tr, 0.1, 0.9)

# Consturcted mu0 and mu1 in OR
clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
clf.fit(x_tr[t_tr == 0], y_tr[t_tr == 0])
y0_or = clf.predict_proba(x_tr)[:, 1]
clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
clf.fit(x_tr[t_tr == 1], y_tr[t_tr == 1])
y1_or = clf.predict_proba(x_tr)[:, 1]

# Consturcted mu0 and mu1 in IPW
clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
clf.fit(x_tr[t_tr == 0], y_tr[t_tr == 0], sample_weight = 1/(1 - p_tr[t_tr == 0]))
y0_ips = clf.predict_proba(x_tr)[:, 1]

clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
clf.fit(x_tr[t_tr == 1], y_tr[t_tr == 1], sample_weight = 1/(p_tr[t_tr == 1]))
y1_ips = clf.predict_proba(x_tr)[:, 1]

# Consturcted mu0 and mu1 in AIPW
mlp = MLP1(input_size = x_tr.shape[1])
mlp.fit(x_tr[t_tr == 0], (y_tr[t_tr == 0]/(1-p_tr[t_tr == 0]) - y0_or[t_tr == 0]/(1-p_tr[t_tr == 0])) + y0_or[t_tr == 0])
y0_dr = mlp.predict(x_tr)
mlp = MLP1(input_size = x_tr.shape[1])
mlp.fit(x_tr[t_tr == 1], (y_tr[t_tr == 1]/(p_tr[t_tr == 1]) - y1_or[t_tr == 1]/(p_tr[t_tr == 1])) + y1_or[t_tr == 1], batch_size = 20)
y1_dr = mlp.predict(x_tr)

with open("data/JOBS/jobs_train_data", "wb") as file:
    pickle.dump(y0_tr, file)
    pickle.dump(y1_tr, file)
    pickle.dump(y0_or, file)
    pickle.dump(y1_or, file)
    pickle.dump(y0_ips, file)
    pickle.dump(y1_ips, file)
    pickle.dump(y0_dr, file)
    pickle.dump(y1_dr, file)
    pickle.dump(p_tr, file)

print("数据已成功保存到文件 'data/JOBS/jobs_train_data'")
print("done")



