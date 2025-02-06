# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch

torch.manual_seed(2020)
from torch import nn

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size, bias = False)
        self.linear_2 = torch.nn.Linear(input_size, input_size // 2, bias = False)
        self.linear_3 = torch.nn.Linear(input_size // 2, 1, bias = False)

    def forward(self, x):

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)

        return torch.squeeze(x)

class MLP1(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, 1, bias = True)
    
    def forward(self, x):
        
        x = self.linear_1(x)     
        x = self.sigmoid(x)
        
        return torch.squeeze(x)      
    
    def fit(self, x, y, num_epoch=1000, lr=0.01, lamb=0, tol=1e-4, batch_size = 20, verbose=True):
        
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
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                
                sub_x = torch.Tensor(x[selected_idx])
                sub_y = torch.Tensor(y[selected_idx])
                
                pred = self.forward(sub_x)

                loss = nn.MSELoss()(pred, sub_y)
                
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().numpy()
                           
            if  epoch_loss > last_loss - tol:
                if early_stop > 5:
                    # print("[IPS_model] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

    def predict(self, x):
        x = torch.Tensor(x)
        x = self.forward(x)
        return x.detach().cpu().numpy()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))



