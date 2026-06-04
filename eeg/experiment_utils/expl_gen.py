import numpy as np
import os
import math
import random
import pickle#5 as pickle
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from captum.attr import Saliency, NoiseTunnel, IntegratedGradients

from model import EEGNet, InterpretableCNN, SCCNet
from model import EEGNet_SSVEP, InterpretableCNN_SSVEP
from utils import train_an_epoch, evaluate_an_epoch, pgd, evaluate_an_epoch_auc, get_loader, getloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_default_dtype(torch.float64) # np float default: float64
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(False)

dataname = 'ERN'
rep = 4

DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/P300"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"
AE_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/ae/{dataname}"
EXPL_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/expl/{dataname}"

ern_subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


# kwerg = dict(n_classes=4,
#             channels=22,
#             samples=562,
#             sfreq=125.0)

kwerg = dict(n_classes=2,
            channels=56,
            samples=160,
            sfreq=128)

# kwerg = dict(n_classes=5,
#             channels=8,
#             samples=125,
#             sfreq=250,)

def get_saliency(model, data_loader): # Gradient
    model.eval()
    expls = None
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()
        
        if expls is None:
            expls= np.zeros((0, x_batch.size(1), x_batch.size(2), x_batch.size(3)))
        x_batch.requires_grad_(True)
        saliency = Saliency(model)
        expl = saliency.attribute(x_batch, target = y_batch, abs = False)#[y_batch == output.argmax(dim=1)]
        expls = np.concatenate((expls, expl.detach().cpu().numpy()))
        
    return expls.squeeze()


def get_saliency_var(model, data_loader, nt_type, bn, nt_samp=25, stdevs=1e-2): # Gradient Variations
    model.eval()
    expls = None
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()

        if expls is None:
            expls= np.zeros((0, x_batch.size(1), x_batch.size(2), x_batch.size(3)))
        x_batch.requires_grad_(True)
        saliency = Saliency(model)
        saliency_sg = NoiseTunnel(saliency)
        expl = saliency_sg.attribute(x_batch, target = y_batch,  abs = False, nt_type=nt_type,nt_samples=nt_samp,
                                     nt_samples_batch_size = bn, stdevs = stdevs)#[y_batch == output.argmax(dim=1)]
        expls = np.concatenate((expls, expl.detach().cpu().numpy()))
        
    return expls.squeeze()


if __name__ == '__main__':
    model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    # model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}
    leg = ['gradient', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'random', 'inte_grad']
    ##================= custom variable initialization
    nts = 16 # 
    bn = 10 
    vsplit = (100, 40) # float or tuple(#train trial, #test trial)
    ##=================
    loss_fn = nn.CrossEntropyLoss()

    for m in model_type.keys():
        if m != 'sccnet':
            continue
        if not os.path.exists(os.path.join(EXPL_DIR, f'{m}')):
            os.makedirs(os.path.join(EXPL_DIR, f'{m}'))
        if not os.path.exists(os.path.join(AE_DIR, f'{m}')):
            os.makedirs(os.path.join(AE_DIR, f'{m}'))


        nsub = 16
        t_acc, a_acc = np.zeros(nsub), np.zeros(nsub)
        for s in range(nsub):
            # sub = s+1
            sub = ern_subs[s]

           
            test_avg, adv_avg = 0,0

            # mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
            mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
            # mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
            xtest, ytest = mat["x_test"][:], mat["y_test"].squeeze()[-40:]
            y_test = torch.Tensor(ytest).long()

            testmodel = model_type[m](**kwerg)
            test_model_path = os.path.join(SAVE_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname, rep, sub, m))
            checkpoint = torch.load(test_model_path, map_location="cpu", weights_only=False)
            testmodel.load_state_dict(checkpoint["state_dict"]) 
            testmodel = testmodel.to(device)

            nstd, _, __, testload = getloader(sub, bn, vsplit=vsplit, dataset=dataname.lower(), datadir=DATASET_DIR)
            test_loss, test_acc, test_out = evaluate_an_epoch_auc(testmodel, device, testload, loss_fn)

            ## =========== adversarial example computation
            # ae = pgd(testmodel, device, testload, loss_fn, epsilon=2e-1)
            # np.save(os.path.join(AE_DIR,f'{m}/sub{sub}'), ae)

            # x_adv = torch.Tensor(ae)
            # advset = torch.utils.data.TensorDataset(x_adv, y_test)
            # advloader = torch.utils.data.DataLoader(advset, batch_size= bn, shuffle=False)
            # # adv_loss, adv_acc, adv_out = evaluate_an_epoch(testmodel, device, advloader, loss_fn)
            # adv_loss, adv_acc, adv_out = evaluate_an_epoch_auc(testmodel, device, advloader, loss_fn)
            
            ## =========== saliency computation
            for l in leg:
                print(sub, m, l)
                if l == 'gradient':
                    grad = get_saliency(testmodel, testloa, abs=False)
                elif l == 'random':
                    expl = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'))
                    random_expl  = np.zeros(expl.shape)
                    for t in range(expl.shape[0]):
                        random_expl[t] = np.random.normal(expl[t].mean(), expl[t].std(), expl[t].shape)
                    grad = random_expl
                elif l =='inte_grad':
                    testmodel.eval()
                    expls = None
                    for i, (x_batch, y_batch) in enumerate(testload):
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        testmodel.zero_grad()                        
                        if expls is None:
                            expls= np.zeros((0, x_batch.size(1), x_batch.size(2), x_batch.size(3)))
                        x_batch.requires_grad_(True)
                        ig = IntegratedGradients(testmodel)
                        expl = ig.attribute(x_batch, target = y_batch)
                        expls = np.concatenate((expls, expl.detach().cpu().numpy()))
                    grad = expls.squeeze()
                else:
                    grad = get_saliency_var(testmodel, testload, l, bn, nts, nstd.item(), abs=False)
                np.save(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), grad)
               