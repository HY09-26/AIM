import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import math
import os
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

def train_an_epoch(model, device, data_loader, loss_fn, optimizer):
    model.train()
    a, b = 0, 0
    epoch_loss = np.zeros((len(data_loader), ))
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss[i] = loss.item()
        b += y_batch.size(0)
        a += torch.sum(y_batch == output.argmax(dim=1)).item()
    return epoch_loss.mean(), a / b

def train_an_epoch_auc(model, device, data_loader, loss_fn, optimizer):
    model.train()
    a, b = np.zeros(0), np.zeros(0)
    epoch_loss = np.zeros((len(data_loader), ))
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss[i] = loss.item()
        b = np.concatenate((b, output[:, 1].detach().cpu().numpy()), 0) # pred
        a = np.concatenate((a, y_batch.detach().cpu().numpy()), 0) # true
    return epoch_loss.mean(), roc_auc_score(a, b)

def evaluate_an_epoch(model,device, data_loader, loss_fn):
    model.eval()
    a, b = 0, 0
    outputs = None
    epoch_loss = np.zeros((len(data_loader), ))
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                                              
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        if outputs is None:
            outputs = np.zeros((0, output.size(-1)))
        outputs = np.concatenate((outputs, output.clone().detach().cpu().numpy()))

        epoch_loss[i] = loss.item()
        b += y_batch.size(0)
        a += torch.sum(y_batch == output.argmax(dim=1)).item()
    return epoch_loss.mean(), a / b, outputs

def evaluate_an_epoch_auc(model,device, data_loader, loss_fn):
    model.eval()
    a, b = np.zeros(0), np.zeros(0)
    outputs = None
    epoch_loss = np.zeros((len(data_loader), ))
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                                              
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        if outputs is None:
            outputs = np.zeros((0, output.size(-1)))
        outputs = np.concatenate((outputs, output.clone().detach().cpu().numpy()))

        epoch_loss[i] = loss.item()
        b = np.concatenate((b, output[:, 1].detach().cpu().numpy()), 0) # pred
        a = np.concatenate((a, y_batch.detach().cpu().numpy()), 0) # true
    return epoch_loss.mean(), roc_auc_score(a, b), outputs

def getloader(sub, bs, vsplit=(100, 40), dataset='ssvep', xtrain=None, ytrain=None, datadir=''):
    """get ern, ssvep dataloaders"""
    if dataset == 'mi':
        if xtrain is None or ytrain is None:
            mat = loadmat(os.path.join(datadir, f"BCIC_S{sub:02d}_T.mat"))
            mat2 = loadmat(os.path.join(datadir, f"BCIC_S{sub:02d}_E.mat"))
            xtrain, ytrain = mat["x_train"], mat["y_train"].squeeze()
            x_test, y_test = mat2["x_test"], mat2["y_test"].squeeze()
        x_train = np.zeros((0, *xtrain.shape[1:]))
        y_train = np.zeros((0, ))
        x_valid = np.zeros((0, *xtrain.shape[1:]))
        y_valid = np.zeros((0, ))
        
        for c in range(4):
            x_, y_ = xtrain[ytrain == c], ytrain[ytrain == c]
            ids = np.random.choice(np.arange(x_.shape[0]), math.floor(x_.shape[0]*vsplit), replace=False)
            mask = np.ones(x_.shape[0], dtype=bool)
            mask[ids]=False
            x_train = np.append(x_train, x_[ids], axis=0)
            y_train = np.append(y_train, y_[ids], axis=0)
            x_valid = np.append(x_valid, x_[mask==True], axis=0)
            y_valid = np.append(y_valid, y_[mask==True], axis=0)

    elif dataset == 'ern':
        if xtrain is None or ytrain is None:
            mat = loadmat(os.path.join(datadir, f"Data_S{sub:02d}_Sess.mat"))
            xtrain, ytrain = mat["x_test"], mat["y_test"].squeeze()
        totaltrials = xtrain.shape[0]
        x_train = xtrain[:totaltrials-(vsplit[0]+vsplit[1])]
        y_train = ytrain[:totaltrials-(vsplit[0]+vsplit[1])]
        x_valid = xtrain[totaltrials-(vsplit[0]+vsplit[1]):totaltrials-(vsplit[1])]
        y_valid = ytrain[totaltrials-(vsplit[0]+vsplit[1]):totaltrials-(vsplit[1])]
        x_test  = xtrain[totaltrials-(vsplit[1]):]
        y_test  = ytrain[totaltrials-(vsplit[1]):]
    elif dataset == 'ssvep':
        if xtrain is None or ytrain is None:
            mat = loadmat(os.path.join(datadir, f"U0{sub:02d}.mat"))
            xtrain, ytrain = mat["x_test"], mat["y_test"].squeeze()
        totaltrials = xtrain.shape[0]
        ids = np.random.choice(np.arange(totaltrials-vsplit[1]), vsplit[0], replace=False)
        mask = np.ones(totaltrials, dtype=bool)
        mask[ids]=False
        mask[totaltrials-vsplit[1]:] = False
        
        x_train = xtrain[mask==True]
        y_train = ytrain[mask==True]
        x_valid = xtrain[ids]
        y_valid = ytrain[ids]
        x_test  = xtrain[totaltrials-(vsplit[1]):]
        y_test  = ytrain[totaltrials-(vsplit[1]):]
    else:
        return 
    
    # print(x_train.shape, y_train.shape)
    # print(x_valid.shape, y_valid.shape)
    # print(x_test.shape, y_test.shape)
    
    x_train, y_train = torch.Tensor(x_train).unsqueeze(1), torch.Tensor(y_train).long()
    x_valid, y_valid = torch.Tensor(x_valid).unsqueeze(1), torch.Tensor(y_valid).long()
    x_test, y_test = torch.Tensor(x_test).unsqueeze(1), torch.Tensor(y_test).long()

    stddev = x_test.mean(axis=0).std()

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    tloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    vloader = torch.utils.data.DataLoader(validset, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    return stddev, tloader, vloader, testloader

def get_loader(bs, xtest=None, ytest=None, sub = 0, datadir=''):
    """get test data stddev & dataloader"""
    if sub !=0 and (xtest is None or ytest is None):
        mat = loadmat(os.path.join(datadir, f"BCIC_S{sub:02d}_E.mat"))
        xtest, ytest = mat["x_test"], mat["y_test"].squeeze()
    x_test, y_test = torch.Tensor(xtest).unsqueeze(1), torch.Tensor(ytest).long()
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

    stddev = x_test.mean(axis=0).std()

    return stddev, testloader


def pgd(model, device, data_loader, loss_fn,
        epsilon = 1e-3, 
        random_start = 1e-8,
        iter_steps = 10,):
    
    eps = epsilon / iter_steps
    model.eval()
    AE = None
    
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_max, x_min = x_batch.max(), x_batch.min()
        y_batch = y_batch.to(device)
        
        if random_start is not None:
            x_batch += torch.rand(x_batch.size()) * random_start
            x_batch = torch.clamp(x_batch, min = x_min, max = x_max)
        
        for it in range(iter_steps):
            x_batch = x_batch.to(device)
            x_batch.requires_grad_()
            
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            
            grad_sign = torch.sign(x_batch.grad.data)
            AP = (grad_sign * eps).detach().cpu()
            
            x_batch = x_batch.detach().cpu()
            x_batch = torch.clamp(x_batch + AP, min = x_min, max = x_max)
        
        #print(f"batch {i} iteration done")
        if AE is None:
            AE = np.zeros((0, *x_batch.size()[1:]))
        AE = np.concatenate((AE, x_batch.detach().cpu().numpy()))
       
    return AE