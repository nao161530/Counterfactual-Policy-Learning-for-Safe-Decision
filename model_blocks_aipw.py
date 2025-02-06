# -*- coding: utf-8 -*-
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
from utils import circle_points, I_fun

mse_func = lambda x, y: np.mean((x - y) ** 2)
acc_func = lambda x, y: np.sum(x == y) / len(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size, bias=False)
        self.linear_2 = torch.nn.Linear(input_size, input_size // 2, bias=False)
        self.linear_3 = torch.nn.Linear(input_size // 2, 1, bias=False)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)

        return torch.squeeze(x)


class model_cate_aipw(nn.Module):
    def __init__(self, input_size):
        super(model_cate_aipw, self).__init__()
        self.input_size = input_size
        self.model = MLP(input_size=self.input_size)
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, t, y, p, mu0, mu1 ,y0_dr, y1_dr, c, rho, num_epoch=50, lr=0.01, lamb=0, tol=1e-4, batch_size=20, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = torch.Tensor(x[selected_idx])
                sub_t = torch.Tensor(t[selected_idx])
                sub_y = torch.Tensor(y[selected_idx])
                sub_p = torch.Tensor(p[selected_idx])
                sub_mu0 = torch.Tensor(mu0[selected_idx])
                sub_mu1 = torch.Tensor(mu1[selected_idx])
                sub_dr0 = torch.Tensor(y0_dr[selected_idx])
                sub_dr1 = torch.Tensor(y1_dr[selected_idx])


                pred = self.model.forward(sub_x)

                loss = torch.sum((pred - 1) * rho * sub_dr0 - pred * (sub_dr1 - c))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().numpy()

            if epoch_loss > last_loss + tol:
                if early_stop > 10:
                    print("[OR_model] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[OR_model] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        x = torch.Tensor(x)
        pred = self.model.forward(x)
        pred = pred.detach().numpy().flatten()
        pred = np.random.binomial(1, pred)
        return pred


class model_lagrange_aipw(nn.Module):
    def __init__(self, input_size):
        super(model_lagrange_aipw, self).__init__()
        self.input_size = input_size
        self.model = MLP(input_size=self.input_size)
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, t, y, p, mu0, mu1, y0_dr, y1_dr, c, rho, harm_num, thr=-5, stop=10, num_epoch=100, batch_size=20, panelty=0, lr=0.01,
            lamb=0,
            tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        num_sample = len(x)
        total_batch = num_sample // batch_size
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = torch.Tensor(x[selected_idx])
                sub_t = torch.Tensor(t[selected_idx])
                sub_y = torch.Tensor(y[selected_idx])
                sub_p = torch.Tensor(p[selected_idx])
                sub_mu0 = torch.Tensor(mu0[selected_idx])
                sub_mu1 = torch.Tensor(mu1[selected_idx])
                sub_dr0 = torch.Tensor(y0_dr[selected_idx])
                sub_dr1 = torch.Tensor(y1_dr[selected_idx])
                pred = self.model.forward(sub_x)

                reward_loss = torch.sum((pred - 1) * rho * sub_dr0 - pred * (sub_dr1 - c))

                harm_loss = torch.sum(((1 - sub_t) * (sub_y - sub_mu0) / (1 - sub_p) + sub_mu0) * pred -
                                      (sub_t * (sub_y - sub_mu1) / sub_p + sub_mu1) * sub_mu0 * pred)
                loss = reward_loss + panelty * torch.max(torch.zeros(1), harm_loss - harm_num / total_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().numpy()

            if epoch % 10 == 0 and verbose:
                print("[OR_model_lagrange] epoch:{}, loss:{}".format(epoch, epoch_loss))

    def predict(self, x):
        x = torch.Tensor(x)
        pred = self.model.forward(x)
        pred = pred.detach().numpy().flatten()
        pred = np.random.binomial(1, pred)
        return pred

class model_constraint_aipw(nn.Module):
    def __init__(self, input_size):
        super(model_constraint_aipw, self).__init__()
        self.input_size = input_size
        self.model = MLP(input_size=self.input_size)
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, t, y, p, mu0, mu1, y0_dr, y1_dr, c, rho, harm_rate, npref, pref_idx, num_epoch=0, batch_size=0, panelty=0,
            lr=0, lamb=0, tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            total_epsilon = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = torch.Tensor(x[selected_idx])
                sub_t = torch.Tensor(t[selected_idx])
                sub_y = torch.Tensor(y[selected_idx])
                sub_p = torch.Tensor(p[selected_idx])
                sub_mu0 = torch.Tensor(mu0[selected_idx])
                sub_mu1 = torch.Tensor(mu1[selected_idx])
                sub_dr0 = torch.Tensor(y0_dr[selected_idx])
                sub_dr1 = torch.Tensor(y1_dr[selected_idx])
                sub_harm_rate = torch.Tensor(harm_rate[selected_idx])

                pred = self.model.forward(sub_x)

                reward_loss = torch.sum((pred - 1) * rho * sub_dr0 - pred * (sub_dr1 - c))

                harm_loss = torch.sum(((1 - sub_t) * (sub_y - sub_mu0) / (1 - sub_p) + sub_mu0) * pred -
                                      (sub_t * (sub_y - sub_mu1) / sub_p + sub_mu1) * sub_mu0 * pred)

                ref_vec = np.array(circle_points([1], [npref])[0])
                beta = ref_vec[pref_idx][1] / ref_vec[pref_idx][0]
                a = np.squeeze(I_fun(sub_dr1 - sub_dr0 - c - beta * sub_harm_rate))
                sub_harm_rate = sub_harm_rate.detach().numpy()
                epsilon = batch_size * np.mean(sub_harm_rate * a)

                loss = reward_loss + panelty * torch.max(torch.zeros(1), harm_loss - epsilon)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_epsilon += epsilon
                epoch_loss += loss.detach().numpy()

            if epoch % 10 == 0:
                print(f'[constriant model] Epoch [{epoch}], epoch_loss:{epoch_loss.item():.4f}')
        return total_epsilon

    def predict(self, x):
        x = torch.Tensor(x)
        pred = self.model.forward(x)
        pred = pred.detach().numpy().flatten()
        pred = np.random.binomial(1, pred)
        return pred

class model_linear_aipw(nn.Module):
    def __init__(self, input_size):
        super(model_linear_aipw, self).__init__()
        self.input_size = input_size
        self.model = MLP(input_size=self.input_size)
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, t, y, p, mu0, mu1, y0_dr, y1_dr, c, rho, npref, pref_idx, num_epoch=0, batch_size=0, lr=0, lamb=0, stop=0, tol=1e-4,
            verbose=True):

        ref_vec = torch.tensor(circle_points([1], [npref])[0]).float().to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        self.model.to(device)
        num_sample = len(x)
        total_batch = num_sample // batch_size
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = torch.Tensor(x[selected_idx]).to(device)
                sub_t = torch.Tensor(t[selected_idx]).to(device)
                sub_y = torch.Tensor(y[selected_idx]).to(device)
                sub_p = torch.Tensor(p[selected_idx]).to(device)
                sub_mu0 = torch.Tensor(mu0[selected_idx]).to(device)
                sub_mu1 = torch.Tensor(mu1[selected_idx]).to(device)
                sub_dr0 = torch.Tensor(y0_dr[selected_idx]).to(device)
                sub_dr1 = torch.Tensor(y1_dr[selected_idx]).to(device)
                pred = self.model.forward(sub_x)

                reward_loss = torch.sum((pred - 1) * rho * sub_dr0 - pred * (sub_dr1 - c))

                harm_loss = torch.sum(((1 - sub_t) * (sub_y - sub_mu0) / (1 - sub_p) + sub_mu0) * pred -
                                      (sub_t * (sub_y - sub_mu1) / sub_p + sub_mu1) * sub_mu0 * pred)

                normalize_coeff = 2 / torch.sum(torch.abs(ref_vec[pref_idx]))
                weight_vec = ref_vec[pref_idx] * normalize_coeff

                loss = weight_vec[0] * reward_loss + weight_vec[1] * harm_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().cpu().numpy()

            if epoch % 10 == 0 and verbose:
                print("[linear_model] epoch:{}, xent:{}".format(epoch, epoch_loss))

    def predict(self, x):
        x = torch.Tensor(x).to(device)
        pred = self.model.forward(x)
        pred = pred.detach().cpu().numpy().flatten()
        pred = np.random.binomial(1, pred)
        return pred
