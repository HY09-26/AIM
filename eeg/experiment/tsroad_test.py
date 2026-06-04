import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from tqdm import tqdm
import mne
import torch
import torch.nn as nn


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--abs_saliency", type=int, choices=[0, 1], required=True, help="1 = absolute saliency")
args = parser.parse_args()

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import train_an_epoch, evaluate_an_epoch, get_loader, evaluate_an_epoch_auc
from experiment_utils.mfbb import MFBB, fit_hurst


os.environ["CUDA_VISIBLE_DEVICES"] ="0"

torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

torch.use_deterministic_algorithms(True)

dataname = 'ERN'
DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/P300"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"


## for MI
# chs = [                  'Fz', 
#            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
#         'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
#            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
#                    'P1', 'Pz', 'P2', 
#                          'POz']
# info = mne.create_info(ch_names = chs, sfreq = 125, ch_types="eeg")

## for ERN
chs = ['Fp1',                             'Fp2',
    'AF7', 'AF3',                      'AF4', 'AF8',
'F7',  'F5',  'F3',  'F1',  'Fz',  'F2',  'F4',  'F6',  'F8',
'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
'T7',  'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',  'T8',
'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
'P7',  'P5',  'P3',  'P1',  'Pz',  'P2',  'P4',  'P6',  'P8',
        'PO7',            'POz',            'PO8',
            'O1',                         'O2']
info = mne.create_info(ch_names = chs, sfreq = 128, ch_types="eeg")

## for SSVEP
# chs = ['PO7', 'PO3', 'O1', 'POz', 'Oz', 'PO4', 'O2', 'PO8']
# info = mne.create_info(ch_names = chs, sfreq = 125, ch_types="eeg")

# hurst = 1e-5


def find_crop(grad, ratio=0.1):
    grad = grad.squeeze()
    if len(grad.shape)>2:
        grad = grad.mean(axis = 0)

    winsize = ratio*grad.shape[-1]
    winsum = np.convolve(grad[0, :], np.ones(int(winsize)), 'valid')

    for c in range(1, grad.shape[0]):
        winsum += np.convolve(grad[c, :], np.ones(int(winsize)), 'valid')

    return winsum.argmax(), winsum.argmin(), int(winsize)


def interp_test(mfbb_dir, model, modelname, benigns,  labels, bn, sub, method, hursts, dir_end=1, mode=('mo',-1), topk=1, start_idx =None, wsize = None):
    benigns = benigns.squeeze() #trial, ch, sample
    mixtures, fbms = benigns.copy(),  benigns.copy()

    (tr, ch, ts) = benigns.shape

    if not os.path.isdir(mfbb_dir+f"//{modelname}//{dir_end}"):
        os.makedirs(mfbb_dir+f"//{modelname}//{dir_end}", exist_ok=True)
    if not os.path.isdir(mfbb_dir+f"//{modelname}//{dir_end}//sub{sub}"):
        os.makedirs(mfbb_dir+f"//{modelname}//{dir_end}//sub{sub}", exist_ok=True)


    if os.path.isfile(os.path.join(mfbb_dir+f"//{modelname}//{dir_end}//sub{sub}", f'sub{sub}_top{topk}_{method}_{mode[0]}.mat')):
        mat = loadmat(os.path.join(mfbb_dir+f"//{modelname}//{dir_end}//sub{sub}", f'sub{sub}_top{topk}_{method}_{mode[0]}.mat'))
        filt = mat['road']

    else:
        Ti = np.array([0, wsize//2, wsize-1]) # Ti = np.array([0, 0.5, 1])
        Xi = benigns[:,:, start_idx+Ti]
        
        for t in range(tr):
            for c in range(ch):
                mfbb, fbm = MFBB(Xi[t, c], Xi[t, c, 2], 1,1e-5, wsize//2, benigns[t, c].std())
                mfbb -= mfbb.mean()
                mixtures[t, c, start_idx:start_idx+wsize] = mfbb
                fbms[t, c, start_idx:start_idx+wsize] = fbm
                ## extended experiment
                # rdn = np.random.normal(benigns[t, c].mean(), benigns[t, c].std(), (1, ts))
                # rdn = np.random.uniform(benigns[t, c].mean(), benigns[t, c].std(), (1, ts))
                # mixtures[t, c, start_idx:start_idx+wsize] = rdn[:, start_idx:start_idx+wsize]

        # post process mfbb
        raw = np.zeros((ch, tr*ts))
        for t in range(tr):
            raw[:, t*ts:(t+1)*ts] = mixtures[t]
        raw = mne.io.RawArray(raw, info, verbose='WARNING')

        ## for MI, ERN, SSVEP respectively
        # raw.filter(l_freq=4, h_freq = 38, verbose='WARNING')
        raw.filter(l_freq=1, h_freq = 40, verbose='WARNING')
        # raw.filter(l_freq=1, h_freq = 50, verbose='WARNING')
        data, times = raw[:]
        filt = benigns.copy()
        for t in range(tr):
            filt[t, :, start_idx:start_idx+wsize] = data[:, t*ts:(t+1)*ts][:, start_idx:start_idx+wsize]

        savemat(os.path.join(mfbb_dir+f"//{modelname}//{dir_end}//sub{sub}", f'sub{sub}_top{topk}_{method}_{mode[0]}_norm.mat'), \
                            dict(road = filt, fbm=fbms, unfilt=mixtures, idx = start_idx, hursts=hursts, win = wsize))
    
    _, test_loader_r = get_loader(bn, filt, labels)
    loss_fn = nn.CrossEntropyLoss()
    # test_loss, test_acc, test_out = evaluate_an_epoch(model,device, test_loader_r, loss_fn)
    test_loss, test_acc, test_out = evaluate_an_epoch_auc(model,device, test_loader_r, loss_fn)
    
    return test_loss, test_acc, test_out


modes = [('mo',-1), ('le',1)]


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

if __name__ == '__main__':
    model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    # model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}

    
    leg = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random']
    filler = '' if args.abs_saliency ==1 else 'n'

    for rep in range(5):
        EXPL_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/expl/{dataname}"
        MFBB_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/mfbbs/{dataname}"

        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_road/{dataname}_sto')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_road/{dataname}_sto'), exist_ok = True)
        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_road/{dataname}')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_road/{dataname}'), exist_ok = True)
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/mfbbs'), exist_ok = True)
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/mfbbs/{dataname}'), exist_ok = True)

        for m in model_type.keys():
            if m != 'eegnet':
                continue

            hists, hists1 = [], []

            for l in leg[:]:
                if args.abs_saliency ==1 and l in ['smoothgrad_sq', 'vargrad']:
                    continue
                testmodel = model_type[m](**kwerg)

                print('rep', rep, m, l, args.abs_saliency)

                dim = 11
                # hist =  dict(acc=np.zeros((9, dim)), loss=np.zeros((9, dim)), out=np.zeros((9, dim, 288, 4)))
                # hist1 = dict(acc=np.zeros((9, dim)), loss=np.zeros((9, dim)), out=np.zeros((9, dim, 288, 4)))
                hist =  dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
                hist1 = dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
                # hist =  dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
                # hist1 = dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
                for s in tqdm(range(16)): #tqdm(range(len(ern_subs))): #tqdm(range(9)): #tqdm(range(11)): # range(11) :# 
                
                    sub = ern_subs[s]
                    # sub = s + 1
                    model_path = os.path.join(SAVE_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname,rep, sub, m))
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    testmodel.load_state_dict(checkpoint["state_dict"]) 
                    testmodel = testmodel.to(device)
                    
                    # mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
                    mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
                    # mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
                    xtest, ytest = mat["x_test"][:], mat["y_test"].squeeze()[:]

                    if l != 'gradientxinput':
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), allow_pickle=True)
                    else:
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'), allow_pickle=True)
                        grads = np.multiply(grads, xtest[-40:])

                    if args.abs_saliency ==1:
                        grads = np.absolute(grads, out=grads)

                    hursts = np.zeros((xtest.shape[0], xtest.shape[1]))


                    
                    for k in range(1, 11):
                        max_idx, min_idx, wsize = find_crop(grads.mean(axis=0), np.round(k*0.05, 2))
                        if wsize %2 !=0:
                            wsize -= 1
                        # print(max_idx, min_idx, wsize)
                        
                        # loss, acc, out = interp_test(MFBB_DIR, testmodel, m, xtest, ytest, 32,\
                        #                                 sub, l, hursts, filler+'abs', modes[0], k, max_idx, wsize)
                        loss, acc, out = interp_test(MFBB_DIR, testmodel,m, xtest[-40:],  ytest[-40:], 32,\
                                                        sub, l, hursts, filler+'abs', modes[0], k, max_idx, wsize)
                        # loss, acc, out = interp_test(MFBB_DIR, testmodel,m, xtest[-100:],  ytest[-100:],25,\
                        #                                 sub, l, hursts, filler+'abs', modes[0], k, max_idx, wsize)
                        hist['acc'][s][k-1] = acc
                        hist['loss'][s][k-1] = loss
                        hist['out'][s][k-1] = out
                        # loss, acc, out = interp_test(MFBB_DIR, testmodel, m, xtest, ytest, 32,\
                        #                                 sub, l, hursts, filler+'abs', modes[1], k,  min_idx, wsize)
                        loss, acc, out = interp_test(MFBB_DIR, testmodel,m, xtest[-40:], ytest[-40:], 32,\
                                                        sub, l, hursts, filler+'abs', modes[1], k,  min_idx, wsize)
                        # loss, acc, out = interp_test(MFBB_DIR, testmodel,m, xtest[-100:],  ytest[-100:], 25,\
                        #                                 sub, l, hursts, filler+'abs', modes[1], k,  min_idx, wsize)
                        hist1['acc'][s][k-1] = acc
                        hist1['loss'][s][k-1] = loss
                        hist1['out'][s][k-1] = out
                
                hists.append(hist)    
                hists1.append(hist1)
            

            with open(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_road/{dataname}_sto/{m}_ts_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
            with open(os.path.join(SAVE_DIR, f'repeat{rep}/ts_test_road/{dataname}_sto/{m}_ts_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists1, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()

            shutil.rmtree(os.path.join(MFBB_DIR, m+'/'+filler+'abs'))