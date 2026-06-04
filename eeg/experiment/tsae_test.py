import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--abs_saliency", type=int, choices=[0, 1], required=True, help="1 = absolute saliency")
args = parser.parse_args()

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import train_an_epoch, evaluate_an_epoch, get_loader, evaluate_an_epoch_auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

torch.use_deterministic_algorithms(True)

dataname = 'MI'
DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/MI"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"

os.makedirs(SAVE_DIR, exist_ok=True)


def find_crop(grad, ratio=0.1):
    grad = grad.squeeze()
    if len(grad.shape)>2:
        grad = grad.mean(axis = 0)

    winsize = ratio*grad.shape[-1]
    winsum = np.convolve(grad[0, :], np.ones(int(winsize)), 'valid')

    for c in range(1, grad.shape[0]):
        winsum += np.convolve(grad[c, :], np.ones(int(winsize)), 'valid')

    return winsum.argmax(), winsum.argmin(), int(winsize)


def interp_test(model, benigns, aes, start_idx, wsize, labels, bn, sub, method):
    benigns, aes = benigns.squeeze(), aes.squeeze() #trial, ch, sample
    mixtures = benigns.copy()
    mixtures[np.arange(benigns.shape[0])[:, None], :,  start_idx:start_idx+wsize+1] = aes[np.arange(benigns.shape[0])[:, None], :, start_idx:start_idx+wsize+1]
    _, test_loader_r = get_loader(bn, mixtures, labels)
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc, test_out = evaluate_an_epoch(model,device, test_loader_r, loss_fn)
    # test_loss, test_acc, test_out = evaluate_an_epoch_auc(model,device, test_loader_r, loss_fn)
    
    return test_loss, test_acc, test_out


modes = [('mo',-1), ('le',1)]

ern_subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


kwerg = dict(n_classes=4,
            channels=22,
            samples=562,
            sfreq=125.0)

# kwerg = dict(n_classes=2,
#             channels=56,
#             samples=160,
#             sfreq=128)

# kwerg = dict(n_classes=5,
#             channels=8,
#             samples=125,
#             sfreq=250,)


if __name__ == '__main__':
    model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    # model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}

    
    leg = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random']
    filler = '' if args.abs_saliency ==1 else 'n'

    for rep in range(5):
        AE_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/ae/{dataname}"
        EXPL_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/expl/{dataname}"

        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_ae/{dataname}')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_ae/{dataname}'))

        for m in model_type.keys():
            # if m != 'sccnet':
            #     continue
            hists, hists1 = [], []
            

            for l in leg[:]:
                if args.abs_saliency ==1 and l in ['smoothgrad_sq', 'vargrad']:
                    continue
                testmodel = model_type[m](**kwerg)

                print('rep', rep, m, l, args.abs_saliency)
                dim = 10
                hist =  dict(acc=np.zeros((9, dim)), loss=np.zeros((9, dim)), out=np.zeros((9, dim, 288, 4)))
                hist1 = dict(acc=np.zeros((9, dim)), loss=np.zeros((9, dim)), out=np.zeros((9, dim, 288, 4)))
                # hist =  dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
                # hist1 = dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
                # hist =  dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
                # hist1 = dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
                for s in tqdm(range(9)): #tqdm(range(len(ern_subs))): #tqdm(range(1,10)):
                    
                    # sub = ern_subs[s]
                    sub = s +1
                    model_path = os.path.join(SAVE_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname, rep, sub, m))
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    testmodel.load_state_dict(checkpoint["state_dict"]) 
                    testmodel = testmodel.to(device)
                    
                    mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
                    # mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
                    # mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
                    xtest, ytest = mat["x_test"], mat["y_test"].squeeze()

                    if l != 'gradientxinput':
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), allow_pickle=True)
                    else:
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'), allow_pickle=True)
                        grads = np.multiply(grads, xtest[:])

                    aes = np.load(os.path.join(AE_DIR, f'{m}/sub{sub}.npy'))
                    
                    
                    for k in range(1,11):
                        if args.abs_saliency:
                            grads = np.absolute(grads)
                        max_idx, min_idx, wsize = find_crop(grads.mean(axis=0), np.round(k*0.05, 2))

                        loss, acc, out = interp_test(testmodel, xtest, aes, max_idx, wsize,  ytest, 32, sub, l)
                        # loss, acc, out = interp_test(testmodel, xtest[-40:], aes[-40:], max_idx, wsize,  ytest[-40:], 32, sub, l)
                        # loss, acc, out = interp_test(testmodel, xtest[-100:], aes[-100:], max_idx, wsize, ytest[-100:], 25, sub, l)
                        hist['acc'][s][k-1] = acc
                        hist['loss'][s][k-1] = loss
                        hist['out'][s][k-1] = out
                        loss, acc, out = interp_test(testmodel, xtest, aes, min_idx, wsize, ytest, 32,sub,l)
                        # loss, acc, out = interp_test(testmodel, xtest[-40:], aes[-40:],  min_idx, wsize,  ytest[-40:], 32, sub, l)
                        # loss, acc, out = interp_test(testmodel, xtest[-100:], aes[-100:],  min_idx, wsize, ytest[-100:], 25, sub, l)
                        hist1['acc'][s][k-1] = acc
                        hist1['loss'][s][k-1] = loss
                        hist1['out'][s][k-1] = out
                hists.append(hist)    
                hists1.append(hist1)
                
            with open(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_ae/{dataname}/{m}_ts_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
            with open(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_ae/{dataname}/{m}_ts_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists1, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()