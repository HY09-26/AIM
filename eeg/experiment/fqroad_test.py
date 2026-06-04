import numpy as np
import os
import shutil
import math
import random
import pickle#5 as pickle
from scipy.io import loadmat, savemat
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.signal as signal
import matplotlib.pyplot as plt
from tqdm import tqdm
# import mne
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

from model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from utils import train_an_epoch, evaluate_an_epoch, get_loader, getloader, evaluate_an_epoch_auc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("absbool", type=int)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"# str(args.absbool)#"0"

# avoid precision error occurence conversion

seed = int(0) #int(np.random.randint(100,size=1)[0]) #7
# print(seed)
np.random.seed(seed)
random.seed(seed)
# np.set_printoptions(precision=8) 

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

torch.use_deterministic_algorithms(True)

dataname = 'SSVEP'
DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/MAMEM"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"



modes = [('mo',-1), ('le',1)]


ern_subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]

def find_neighbors(den, grad, ratio, absbool, mode):
    # den shape: ch, fbin
    # ratio: 0~1 float
    den_avg = abs(den).mean(axis=0) if len(den.shape) >1 else abs(den)
    grad_avg = abs(grad).mean(axis=0) if len(grad.shape) >1 else abs(grad)
    den_len = den_avg.shape[-1]

    target = den_avg.sum() * ratio
    target_hv = target/2

    accus = np.ones((den_len, 4))*-den_len # lsum, lid, rsum,  rid 
    accus[:, 0], accus[:,2] = den_avg, den_avg
    accus[:,1][den_avg>=target] = 0
    accus[:,3][den_avg>=target] = 0
    grad_accu =  np.zeros((den_len, 2)) # lsum(include self)
    grad_accu[:, 0] = grad_avg

    for i in range(den_len-1):
        lsum_short = np.logical_and(accus[i+1:,0]<target_hv, accus[i+1:,1]<0)
        accus[i+1:, 0][lsum_short] += den_avg[:den_len-i-1][lsum_short]
        grad_accu[i+1:, 0][lsum_short] += grad_avg[:den_len-i-1][lsum_short]
        accus[:,1][np.logical_and(accus[:,0]>=target_hv,accus[:, 1]<0)] = i+1
        
        rsum_short = np.logical_and(accus[:den_len-i-1,2]<target_hv, accus[:den_len-i-1,3]<0)
        accus[:den_len-i-1, 2][rsum_short] += den_avg[i+1:][rsum_short]
        grad_accu[:den_len-i-1, 1][rsum_short] += grad_avg[i+1:][rsum_short]
        accus[:,3][np.logical_and(accus[:,2]>=target_hv, accus[:, 3]<0)] = i+1
    

    valid_lr = np.logical_and(accus[:, 3]>=0, accus[:, 1]>=0)


    neighborhood = np.zeros(den_len)
    neighborhood[valid_lr] = grad_accu.sum(axis=-1)[valid_lr]/(accus[valid_lr, 1]+accus[valid_lr, 3]+1)#density

    invalid_idl =  np.arange(den_len)[(accus[:,1]<0)]
    invalid_idr =  np.arange(den_len)[(accus[:,3]<0)]

    for il in invalid_idl:
        ll = 1
        while accus[il, 0]<target and il+ll<den_len:
            accus[il, 0] = den_avg[:il+ll].sum()
            accus[il, 1] = ll*-1
            grad_accu[il, 0] = grad_avg[:il+ll].sum()
            ll += 1
            
    for ir in invalid_idr:
        rr = 1
        while accus[ir, 2]<target and rr<=ir:
            accus[ir, 2] = den_avg[ir-rr:].sum()
            accus[ir, 3] = rr * -1
            grad_accu[ir, 1] = grad_avg[ir-rr:].sum()
            rr += 1
            

    if np.logical_or(accus[:, 3]<den_len*-1, accus[:, 0]<den_len*-1).sum()>0:
        print('warn', (accus[:, 3]<den_len*-1).sum(), (accus[:, 0]<den_len*-1).sum())


    for il in invalid_idl:
        neighborhood[il] = grad_accu[il, 0] / (accus[il, 1]*-1 + il+1)
    for ir in invalid_idr:
        neighborhood[ir] = grad_accu[ir, 1] / (accus[ir, 3]*-1 + den_len-ir)

    
    if mode =='mo':
        m_id = np.array(neighborhood).argmax()
    else:
        m_id = np.array(neighborhood).argmin()



    if accus[m_id, 1]<0:
        return m_id+1, (1, m_id-int(accus[m_id, 1])+1), 0
    elif accus[m_id, 3]<0:
        return m_id+1, (m_id+int(accus[m_id, 3])+1, den_len), 1
    else:
        return m_id+1, (m_id-int(accus[m_id, 1])+1, m_id+int(accus[m_id, 3])+1), 2
    # return frequency center index, left index, right index, return case(for checking)

sqrt2 = np.sqrt(2)

def freq_interp_test(spint_dir, model, modelname, benign, benign_fq, grad_fq, f, lowfq, highfq,  k,  bn, labels, sub=1, method='gradient', absbool = 1, mode = modes[0]): # 22, 562  
    mixture_fq = benign_fq.copy()
    

    dir_str = 'abs' if absbool == 1 else 'nabs'

    if not os.path.isdir(spint_dir+f"//{modelname}"):
        os.makedirs(spint_dir+f"//{modelname}", exist_ok=True)
    if not os.path.isdir(spint_dir+f"//{modelname}//{dir_str}"):
        os.makedirs(spint_dir+f"//{modelname}//{dir_str}", exist_ok=True)


    if os.path.isfile(os.path.join(spint_dir, f'{modelname}//{dir_str}//sub{sub}_top{k:02d}_{method}_{mode[0]}.mat')):#  and method != 'random':
        mat = loadmat(os.path.join(spint_dir, f'{modelname}//{dir_str}//sub{sub}_top{k:02d}_{method}_{mode[0]}.mat'))
        mixture = mat['road']
    else:
        (tr, ch, ts) =  benign.shape

        f = np.fft.fftfreq(ts, d = 1/sfreq)
        uq_freqs = np.where(np.isclose(f, -1*(sfreq/2), 0.5))[0][0]
        neighbors = np.zeros((tr, 3))
        fits = np.zeros((tr, 4))

        for t in range(tr):
            fs_grad = grad_fq[t]
            cid, neighbor_tup, tt = find_neighbors(benign_fq[t, :, 1:uq_freqs], abs(fs_grad[:, 1:uq_freqs]), k*0.01, absbool, mode[0])
            neighbors[t] = [cid, neighbor_tup[0], neighbor_tup[1]]
        
            # =================================
            def overf(x, aa, bb, cc, dd):
                yfit = aa * x**3 + bb * x**2 + cc * x**1 + dd 
                if f[neighbor_tup[0]] >0:
                    yfit[x==1/f[neighbor_tup[0]]] = abs(fs_grad[:, neighbor_tup[0]]).mean(axis=0)/sqrt2
                else:
                    yfit[x>1/f[lowfq]] = abs(fs_grad[:, lowfq]).mean(axis=0)/sqrt2
                yfit[x==1/f[neighbor_tup[1]]] = abs(fs_grad[:, neighbor_tup[1]]).mean(axis=0)/sqrt2
                return yfit
            
            popt, pcov = curve_fit(overf, 1/f[lowfq:uq_freqs], abs(benign_fq[t, :, lowfq:uq_freqs]).mean(axis=0)/sqrt2)
            fits[t] = popt
            powlaw = np.poly1d(popt) # P(1/f)
            # =================================

            impt = np.zeros(benign_fq[t, :, neighbor_tup[0]:neighbor_tup[1]+1].mean(axis=0).shape)
            mu, sigma = (abs(benign_fq[t, :, neighbor_tup[0]:neighbor_tup[1]+1].mean(axis=0))/sqrt2).mean(),\
                        (abs(benign_fq[t, :, neighbor_tup[0]:neighbor_tup[1]+1].mean(axis=0))/sqrt2).std()
            
            for i in range(neighbor_tup[1]+1-neighbor_tup[0]):
                if neighbor_tup[0]+i <lowfq:
                    impt[i] = powlaw(1/f[lowfq])
                else:
                    impt[i:] = powlaw(1/f[neighbor_tup[0]+i:neighbor_tup[1]+1])
                    break

            impt_r, impt_i = np.repeat(impt[None, :], ch, axis=0), np.repeat(impt[None, :], ch, axis=0)
            for c in range (ch):
                impt_r[c] *= np.sign(benign_fq[t, c, neighbor_tup[0]:neighbor_tup[1]+1].real)
                impt_i[c] *= np.sign(benign_fq[t, c, neighbor_tup[0]:neighbor_tup[1]+1].imag)
            

        
            mixture_fq[t, :, neighbor_tup[0]:neighbor_tup[1]+1] = impt_r + 1j*impt_i
            
            if neighbor_tup[0]>0:
                mixture_fq[t, :, ts-neighbor_tup[1]:ts-neighbor_tup[0]+1] = np.flip(impt_r, axis = -1) - 1j*impt_i
            else:
                mixture_fq[t, :, ts-neighbor_tup[1]:ts-neighbor_tup[0]+1] = np.flip(impt_r, axis = -1)[:, :neighbor_tup[1]-neighbor_tup[0]] - 1j*impt_i[:, :neighbor_tup[1]-neighbor_tup[0]]
                            

            
        mixture =  np.fft.ifft(mixture_fq).real
        savemat(os.path.join(spint_dir, f'{modelname}//{dir_str}//sub{sub}_top{k:02d}_{method}_{mode[0]}.mat'), \
                dict(road = mixture, neighbor = neighbors, fit = fits))
                # dict(road = mixture))

    _, test_loader_r = get_loader(bn, mixture, labels)
    loss_fn = nn.CrossEntropyLoss()
    losses, accs, outs = evaluate_an_epoch(model, device, test_loader_r, loss_fn)
    # losses, accs, outs = evaluate_an_epoch_auc(model, device, test_loader_r, loss_fn)
    
    return losses, accs, outs

## for MI
# kwerg = dict(n_classes=4,
#             channels=22,
#             samples=562,
#             sfreq=125.0)
# sfreq = 125

## for ERN
# kwerg = dict(n_classes=2,
#             channels=56,
#             samples=160,
#             sfreq=128)
# sfreq = 128

## for SSVEP
kwerg = dict(n_classes=5,
            channels=8,
            samples=125,
            sfreq=250,)
sfreq = 125

if __name__ == '__main__':
    
    # model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}
    
    leg = ['gradient', 'gradientxinput','smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random',  ]
    filler = '' if args.absbool ==1 else 'n'
    for rep in range(1,3): # 0: 1,2 
        AE_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/ae/{dataname}"
        EXPL_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/expl/{dataname}"

        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/{dataname}')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/{dataname}'))
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/res_data'), exist_ok = True)
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/res_data/{dataname}'), exist_ok = True)
        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/{dataname}_si')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/{dataname}_si'))
        SPINT_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/fq_test_road/res_data/{dataname}"
        for m in model_type.keys():
            # if m != 'eegnet':
            #     continue

            hists, hists1 = [], []


            for l in leg[:]:
                if args.absbool ==1 and l in ['smoothgrad_sq', 'vargrad']:
                    continue
                    
                testmodel = model_type[m](**kwerg)

                dim = 10
                # hist =  dict(acc=np.zeros((9, dim)), loss=np.zeros((9,dim)), out=np.zeros((9, dim, 288, 4)))
                # hist1 = dict(acc=np.zeros((9, dim)), loss=np.zeros((9,dim)), out=np.zeros((9, dim, 288, 4)))
                # hist =  dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
                # hist1 = dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
                hist =  dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
                hist1 = dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
                print('rep', rep, m, l)

                for s in tqdm(range(11)): #tqdm(range(9)): #tqdm(range(11)): #tqdm(range(len(ern_subs))): # 
                    
                    # sub = ern_subs[s]
                    sub = s+1
                    model_path = os.path.join(SAVE_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname,rep, sub, m))
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    testmodel.load_state_dict(checkpoint["state_dict"]) 
                    testmodel = testmodel.to(device)
                    
                    # mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
                    # mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
                    mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
                    xtest, ytest = mat["x_test"][-100:], mat["y_test"].squeeze()[-100:]

                    if l != 'gradientxinput':
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), allow_pickle=True)
                    else:
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'), allow_pickle=True)
                        grads = np.multiply(grads, xtest)

                    if args.absbool == 1:
                        grads = np.absolute(grads, out = grads)

                    freqs = np.fft.fftfreq(grads.shape[-1], d = 1/sfreq)
                    ## for normalizing DC term
                    # dc, gdc = xtest.mean(axis=-1)[:,:, None], grads.mean(axis=-1)[:,:, None]
                    # dc, gdc = np.repeat(dc, xtest.shape[-1], axis = -1), np.repeat(gdc, grads.shape[-1], axis=-1)
                    # benign_norm = xtest-dc
                    # grads_norm = grads-gdc
                    benign_fq = np.fft.fft(xtest)
                    grad_fq = np.fft.fft(grads)
                    for k in range(1,11): #tqdm(range(1,11)):
                    
                        # loss, acc, out = freq_interp_test(SPINT_DIR,testmodel, m, xtest, benign_fq, grad_fq, freqs, 19, 176, k*5, \
                        #                                    32, ytest, sub, l, args.absbool,  mode = modes[0])
                        # loss, acc, out = freq_interp_test(SPINT_DIR,testmodel, m, xtest, benign_fq, grad_fq, freqs, 1, 51, k*5,\
                        #                                      32, ytest, sub, l, args.absbool, mode = modes[0])
                        loss, acc, out = freq_interp_test(SPINT_DIR, testmodel, m, xtest, benign_fq, grad_fq, freqs, 1,41, k*5,\
                                                             25, ytest, sub, l, args.absbool, mode = modes[0])
                        hist['acc' ][s][k-1] = acc
                        hist['loss'][s][k-1] = loss
                        hist['out' ][s][k-1] = out
                        # loss, acc, out = freq_interp_test(SPINT_DIR,testmodel, m, xtest, benign_fq, grad_fq, freqs, 19, 176, k*5,\
                        #                                     32, ytest, sub, l, args.absbool, mode = modes[1])
                        # loss, acc, out = freq_interp_test(SPINT_DIR,testmodel, m, xtest, benign_fq, grad_fq, freqs, 1, 51, k*5,
                        #                                      32, ytest, sub, l, args.absbool, mode = modes[1])
                        loss, acc, out = freq_interp_test(SPINT_DIR, testmodel, m, xtest, benign_fq, grad_fq, freqs, 1,41, k*5,\
                                                             25, ytest, sub, l, args.absbool, mode = modes[1])
                        hist1['acc' ][s][k-1] = acc
                        hist1['loss'][s][k-1] = loss
                        hist1['out' ][s][k-1] = out
                hists.append(hist)    
                hists1.append(hist1)
                



            with open(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/{dataname}_si/{m}_fq_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
            with open(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_road/{dataname}_si/{m}_fq_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists1, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
            shutil.rmtree(os.path.join(SPINT_DIR, m))