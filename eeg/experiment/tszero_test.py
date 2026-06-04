import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import random
import pickle
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("absbool", type=int)
args = parser.parse_args()

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import train_an_epoch, evaluate_an_epoch, get_loader, evaluate_an_epoch_auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# ===== CHANGE ONLY THIS LINE =====
dataname = 'SSVEP'   # 'MI', 'ERN', or 'SSVEP'
# =================================

# ===== PATH CONFIGURATION =====
MY_SAVE_DIR   = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"
IRISHSIEH_DIR = "/mnt/right/alumni/2023/irishsieh/atk"
# ==============================

if dataname == 'MI':
    DATASET_DIR    = "/mnt/right/alumni/2023/irishsieh/datasets/MI"
    kwerg          = dict(n_classes=4, channels=22, samples=562, sfreq=125.0)
    use_ssvep_model= False
    sub_list       = list(range(1, 10))
    n_trials       = 288
    n_classes_out  = 4
    xtest_sl       = slice(None)
    gradxi_sl      = slice(None)
    use_auc        = False
    batch_size     = 32
    def mat_file(s): return f"BCIC_S{s:02d}_E.mat"

elif dataname == 'ERN':
    DATASET_DIR    = "/mnt/right/alumni/2023/irishsieh/datasets/P300"
    kwerg          = dict(n_classes=2, channels=56, samples=160, sfreq=128)
    use_ssvep_model= False
    sub_list       = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    n_trials       = 40
    n_classes_out  = 2
    xtest_sl       = slice(-40, None)
    gradxi_sl      = slice(-40, None)
    use_auc        = True
    batch_size     = 32
    def mat_file(s): return f"Data_S{s:02d}_Sess.mat"

elif dataname == 'SSVEP':
    DATASET_DIR    = "/mnt/right/alumni/2023/irishsieh/datasets/MAMEM"
    kwerg          = dict(n_classes=5, channels=8, samples=125, sfreq=250)
    use_ssvep_model= True
    sub_list       = list(range(1, 12))
    n_trials       = 100
    n_classes_out  = 5
    xtest_sl       = slice(-100, None)
    gradxi_sl      = slice(-100, None)
    use_auc        = False
    batch_size     = 25
    def mat_file(s): return f"U0{s:02d}.mat"

SAVE_DIR = MY_SAVE_DIR
modes    = [('mo', -1), ('le', 1)]
n_subs   = len(sub_list)
ts_dim   = 11   # k in range(1,12), step = k*5% of time window


def find_crop(grad, ratio=0.1):
    grad = grad.squeeze()
    if len(grad.shape) > 2:
        grad = grad.mean(axis=0)
    winsize = ratio * grad.shape[-1]
    winsum  = np.convolve(grad[0, :], np.ones(int(winsize)), 'valid')
    for c in range(1, grad.shape[0]):
        winsum += np.convolve(grad[c, :], np.ones(int(winsize)), 'valid')
    return winsum.argmax(), winsum.argmin(), int(winsize)


def interp_test(model, modelname, benigns, labels, bn, sub, method,
                start_idx=None, wsize=None, mode=('mo', -1), topk=1):
    """Zero-masking: replace selected time window with 0."""
    benigns  = benigns.squeeze()
    mixtures = benigns.copy()
    mixtures[:, :, start_idx:start_idx + wsize] = 0.0
    _, test_loader_r = get_loader(bn, mixtures, labels)
    loss_fn = nn.CrossEntropyLoss()
    if use_auc:
        test_loss, test_acc, test_out = evaluate_an_epoch_auc(model, device, test_loader_r, loss_fn)
    else:
        test_loss, test_acc, test_out = evaluate_an_epoch(model, device, test_loader_r, loss_fn)
    return test_loss, test_acc, test_out


if __name__ == '__main__':
    if use_ssvep_model:
        model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet': SCCNet}
    else:
        model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet': SCCNet}

    leg    = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random']
    filler = '' if args.absbool == 1 else 'n'

    for rep in range(1):
        EXPL_DIR = f"{IRISHSIEH_DIR}/repeat{rep}/expl/{dataname}"
        os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_zero/{dataname}_sto'), exist_ok=True)

        for m in model_type.keys():
            #if m != 'eegnet':
            #   continue

            hists, hists1 = [], []

            for l in leg:
                if args.absbool == 1 and l in ['smoothgrad_sq', 'vargrad']:
                    continue
                testmodel = model_type[m](**kwerg)
                print('rep', rep, m, l, args.absbool)

                hist  = dict(acc =np.zeros((n_subs, ts_dim)),
                             loss=np.zeros((n_subs, ts_dim)),
                             out =np.zeros((n_subs, ts_dim, n_trials, n_classes_out)))
                hist1 = dict(acc =np.zeros((n_subs, ts_dim)),
                             loss=np.zeros((n_subs, ts_dim)),
                             out =np.zeros((n_subs, ts_dim, n_trials, n_classes_out)))

                for s in tqdm(range(n_subs)):
                    sub = sub_list[s]

                    model_path = os.path.join(IRISHSIEH_DIR,
                        f"models/{dataname}/bests_repeat{rep}/sub{sub}-{m}.pth")
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    testmodel.load_state_dict(checkpoint["state_dict"])
                    testmodel = testmodel.to(device)

                    mat        = loadmat(os.path.join(DATASET_DIR, mat_file(sub)))
                    xtest_full = mat["x_test"]
                    ytest_full = mat["y_test"].squeeze()
                    xtest      = xtest_full[xtest_sl]
                    ytest      = ytest_full[xtest_sl]

                    if l != 'gradientxinput':
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'),
                                        allow_pickle=True)
                    else:
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'),
                                        allow_pickle=True)
                        grads = np.multiply(grads, xtest_full[gradxi_sl])

                    if args.absbool == 1:
                        grads = np.absolute(grads, out=grads)

                    for k in range(1, ts_dim + 1):
                        max_idx, min_idx, wsize = find_crop(grads.mean(axis=0), np.round(k * 0.05, 2))
                        if wsize % 2 != 0:
                            wsize -= 1

                        loss, acc, out = interp_test(
                            testmodel, m, xtest, ytest, batch_size,
                            sub, l, max_idx, wsize, modes[0], k)
                        hist['acc'][s][k-1]  = acc
                        hist['loss'][s][k-1] = loss
                        hist['out'][s][k-1]  = out

                        loss, acc, out = interp_test(
                            testmodel, m, xtest, ytest, batch_size,
                            sub, l, min_idx, wsize, modes[1], k)
                        hist1['acc'][s][k-1]  = acc
                        hist1['loss'][s][k-1] = loss
                        hist1['out'][s][k-1]  = out

                hists.append(hist)
                hists1.append(hist1)

            with open(os.path.join(SAVE_DIR,
                    f'repeat{rep}/ts_test_zero/{dataname}_sto/{m}_ts_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as f:
                pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(SAVE_DIR,
                    f'repeat{rep}/ts_test_zero/{dataname}_sto/{m}_ts_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as f:
                pickle.dump(hists1, f, protocol=pickle.HIGHEST_PROTOCOL)