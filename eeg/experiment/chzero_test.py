import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("absbool", type=int)
parser.add_argument("rep", type=int)
args = parser.parse_args()


from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from  experiment_utils.utils import train_an_epoch, evaluate_an_epoch, evaluate_an_epoch_auc, get_loader, getloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0" ## adjust according to server
rep = args.rep

torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

torch.use_deterministic_algorithms(True)

dataname = 'ERN' ## adjust location accordingly

# ===== PATH CONFIGURATION =====
MY_SAVE_DIR   = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"  
IRISHSIEH_DIR = "/mnt/right/alumni/2023/irishsieh/atk"         
DATASET_DIR   = "/mnt/right/alumni/2023/irishsieh/datasets/P300"  
# ==============================

SAVE_DIR  = MY_SAVE_DIR   
AE_DIR    = f"{IRISHSIEH_DIR}/repeat{rep}/ae/{dataname}"
EXPL_DIR  = f"{IRISHSIEH_DIR}/repeat{rep}/expl/{dataname}"
os.makedirs(SAVE_DIR, exist_ok=True)

modes = [('mo',-1), ('le',1)]

ern_subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


def interp_test(model, modelname, benigns, chrank, labels, bn, sub, method):
    """Zero-masking: replace selected channels with 0."""
    benigns = benigns.squeeze()
    mixtures = benigns.copy()

    for t in range(benigns.shape[0]):
        mixtures[t, chrank[t]] = 0.0

    _, test_loader_r = get_loader(bn, mixtures, labels)
    loss_fn = nn.CrossEntropyLoss()
    ## for MI, SSVEP
    # test_loss, test_acc, test_out = evaluate_an_epoch(model, device, test_loader_r, loss_fn)
    ## for ERN
    test_loss, test_acc, test_out = evaluate_an_epoch_auc(model, device, test_loader_r, loss_fn)

    return test_loss, test_acc, test_out, mixtures

## for MI
# kwerg = dict(n_classes=4,
#             channels=22,
#             samples=562,
#             sfreq=125.0)

## for ERN
kwerg = dict(n_classes=2,
            channels=56,
            samples=160,
            sfreq=128)

## for SSVEP
# kwerg = dict(n_classes=5,
#             channels=8,
#             samples=125,
#             sfreq=250,)

if __name__ == '__main__':
    ## for MI, ERN
    model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    ## for SSVEP
    # model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}

    leg = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random']
    for m in model_type.keys():
        filler = '' if args.absbool ==1 else 'n'
        hists, hists1 = [], []

        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_zero/{dataname}')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_zero/{dataname}'))

        for l in leg:
            if args.absbool ==1 and l in ['smoothgrad_sq', 'vargrad']:
                continue
            testmodel = model_type[m](**kwerg)

            print(m, l)
            # adjust according to dataset
            dim = 56
            ## for MI
            # hist  = dict(acc=np.zeros((9, dim)), loss=np.zeros((9,dim)), out=np.zeros((9, dim, 288, 4)))
            # hist1 = dict(acc=np.zeros((9, dim)), loss=np.zeros((9,dim)), out=np.zeros((9, dim, 288, 4)))
            ## for ERN
            hist =  dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
            hist1 = dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
            ## for SSVEP
            # hist =  dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
            # hist1 = dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
            # for s in tqdm(range(11)): 
            for s in tqdm(range(len(ern_subs))): ## change accordingly

                ## for ERN
                sub = ern_subs[s]
                ## for MI, SSVEP
                # sub = s+1

                model_path = os.path.join(IRISHSIEH_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname, rep, sub, m))
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                testmodel.load_state_dict(checkpoint["state_dict"])
                testmodel = testmodel.to(device)

                ## for MI
                # mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
                ## for ERN
                mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
                ## for SSVEP
                # mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
                xtest, ytest = mat["x_test"], mat["y_test"].squeeze()
                if l != 'gradientxinput':
                    grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), allow_pickle=True)
                else:
                    grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'), allow_pickle=True)
                    ## for MI, ERN & SSVEP respectively
                    # grads = np.multiply(grads, xtest[:])
                    grads = np.multiply(grads, xtest[-40:])
                    # grads = np.multiply(grads, xtest[-100:])

                if args.absbool:
                    grads = np.absolute(grads)

                for k in range(1, xtest.shape[1]+1):
                    chrankM = np.argsort(grads.sum(axis=-1)*modes[0][1], axis=-1)[:, :k+1]
                    chrankL = np.argsort(grads.sum(axis=-1)*modes[1][1], axis=-1)[:, :k+1]

                    ## for MI, ERN & SSVEP respectively
                    # loss, acc, out, res = interp_test(testmodel, m, xtest, chrankM,  ytest, 32, sub, l)
                    loss, acc, out, res = interp_test(testmodel, m, xtest[-40:], chrankM,  ytest[-40:], 25, sub, l)
                    # loss, acc, out, res = interp_test(testmodel, m, xtest[-100:], chrankM,  ytest[-100:], 25, sub, l)
                    hist['acc'][s][k-1] = acc
                    hist['loss'][s][k-1] = loss
                    hist['out'][s][k-1] = out

                    ## for MI, ERN & SSVEP respectively
                    # loss, acc, out, res = interp_test(testmodel, m, xtest, chrankL,  ytest, 32, sub, l)
                    loss, acc, out, res = interp_test(testmodel, m, xtest[-40:], chrankL,  ytest[-40:], 25, sub, l)
                    # loss, acc, out, res = interp_test(testmodel, m, xtest[-100:], chrankL,  ytest[-100:], 25, sub, l)
                    hist1['acc'][s][k-1] = acc
                    hist1['loss'][s][k-1] = loss
                    hist1['out'][s][k-1] = out
            hists.append(hist)
            hists1.append(hist1)

        filler = '' if args.absbool ==1 else 'n'

        with open(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_zero/{dataname}/{m}_ch_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as handle:
            pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
        with open(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_zero/{dataname}/{m}_ch_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as handle:
            pickle.dump(hists1, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()