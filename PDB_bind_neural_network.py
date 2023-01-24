# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:54:15 2023

@author: Emily
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class DeepLearn(nn.Module):
    
    # model works for layers of any same size and up to 3 layers
    def __init__(self,size=256,layers=1):
        
        super(DeepLearn, self).__init__()
        self.layer = layers # for using in forward function
        self.fc1 = nn.Linear(2048,size) 
        
        if layers >= 2: # wasnt sure if unused layers would cause problems, so I put them behind ifs
        
            self.fc2 = nn.Linear(size,size)
            
        if layers == 3:
            
            self.fc3 = nn.Linear(size,size)
            
        self.fcf = nn.Linear(size,1)
        
    def forward(self, x):
        
        x = self.fc1(x)
        
        if self.layer >= 2:
            
            x = F.relu(x)
            x = self.fc2(x)
            
        if self.layer == 3:
            
            x = F.relu(x)
            x = self.fc3(x)
            
        x = F.relu(x)
        x = self.fcf(x)
        
        return x

class Data(Dataset):
    
    def __init__(self,path):
        
        self.df = pd.read_csv(path)
        self.input = self.df[self.df.columns[0:-1]].values
        self.targets = self.df[self.df.columns[-1]].values
    
    def __len__(self):
        
        return len(self.targets)
    
    def __getitem__(self, index):
        
        x = self.input[index]
        y = self.targets[index]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train(model,device,train_dl,optim):
    
    model.train()
    loss_fn = torch.nn.L1Loss(reduction="sum")
    loss_tot = 0
    
    for b_i, (f_p, z) in enumerate(train_dl):
        
        f_p, z = f_p.to(device), z.to(device)
        optim.zero_grad()
        pred_prob = model(f_p)
        loss = loss_fn(pred_prob, z.view(-1,1))
        loss.backward()
        optim.step()
        loss_tot += loss.item()
    
    loss_tot /= len(train_dl.dataset)
    
    return loss_tot

def validate(model, device, valid_dl, epoch):
    
    model.eval()
    loss_fn = torch.nn.L1Loss(reduction="sum")
    loss_tot = 0
    
    with torch.no_grad():
        
        for f_p, z in valid_dl:
            
            f_p, z = f_p.to(device), z.to(device)
            pred_prob = model(f_p)
            loss_tot += loss_fn(pred_prob, z.view(-1,1)).item()
            
    loss_tot /= len(valid_dl.dataset)
    print("\nEpoch:{}   Validation dataset: Loss per Datapoint: {:.4f}".format(epoch+1, loss_tot)) #to track progress
    
    return loss_tot

def predict(model, device, dataloader):
    
    model.eval()
    x_all = []
    y_all = []
    pred_prob_all = []
    
    with torch.no_grad():
        
        for m, n in dataloader:
            
            m, n = m.to(device), n.to(device)
            pred_prob = model(m)
            x_all.append(m)
            y_all.append(n)
            pred_prob_all.append(pred_prob)
            
    return (torch.concat(x_all),torch.concat(y_all),torch.concat(pred_prob_all).view(-1))


# can edit values to the following variable matrices to test different hyperparameter combinations
# example: s_layer = [128, 32, 256]

number_layers = [2] 
size_layers = [256] 
size_batches = [32] 
learning_rate = [1e-3] 
n_epoch = [10] 

for a in number_layers:
    
    for b in size_layers:
        
        for c in size_batches:
            
            for d in learning_rate:
                
                for e in n_epoch:
                    
                    # import data - important to change pathway
                    data = Data('pdbind_full_fp2.csv') 
                    
                    # train and test splits
                    n_train = int(len(data) * 0.8)
                    n_val = len(data) - n_train
                    train_set, val_set = torch.utils.data.random_split(data, [n_train, n_val], generator=torch.Generator().manual_seed(0))
                    
                    # preparing model
                    torch.manual_seed(0)
                    train_dl = DataLoader(train_set, batch_size=c, shuffle=True)
                    val_dl = DataLoader(val_set, batch_size=c, shuffle=True)
                    device = torch.device('cpu')
                    model = DeepLearn(b,a)
                    model.to(device)
                    optimizer = optim.Adadelta(model.parameters(),lr=d)
                    
                    # for measuring loss in each epoch
                    train_losses = []
                    val_losses = []
                    
                    for epoch in range(e):
                        
                        train_loss = train(model, device, train_dl, optimizer)
                        train_losses.append(train_loss)
                    
                        validation_loss = validate(model, device, val_dl, epoch)
                        val_losses.append(validation_loss)
                    
                    # plot of losses
                    x_axs = range(1,len(train_losses)+1)
                    plt.plot(x_axs,train_losses,'k',x_axs,val_losses,'r--') 
                    plt.xlabel("Epoch")
                    plt.ylabel("Total Loss / Batch Size")
                    plt.show()
                    
                    x_all, y_all, pred_prob_all = predict(model,device,val_dl)
                    
                    r2 = r2_score(y_all, pred_prob_all)
                    mae = mean_absolute_error(y_all, pred_prob_all)
                    rmse = mean_squared_error(y_all, pred_prob_all, squared=False)
                    
                    # prediction plot
                    plt.figure(figsize=(4, 4), dpi=100)
                    plt.title("Number of layers: {}   Size of layers: {}\nSize of batches: {}   Learning Rate: {}\nNumber of Epochs: {}".format(a,b,c,d,e))
                    plt.scatter(y_all, pred_prob_all, alpha=0.3)
                    plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], color="k", ls="--")
                    plt.figtext(0.15,-0.05,'$R^2: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}$'.format(r2,mae,rmse))
                    plt.xlabel("True Values")
                    plt.ylabel("Predicted Values")
                    plt.show()




