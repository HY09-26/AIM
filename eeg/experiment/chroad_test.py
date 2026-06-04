import numpy as np
import os
import math
import random
import pickle#5 as pickle
from scipy.io import loadmat, savemat
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
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


from model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from utils import train_an_epoch, evaluate_an_epoch, evaluate_an_epoch_auc, get_loader, getloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0" ## adjust according to server
rep = args.rep

torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

torch.use_deterministic_algorithms(True)

dataname = 'MI' ## adjust location accordingly
DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/MI"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"
AE_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/ae/{dataname}"
EXPL_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/expl/{dataname}"
os.makedirs(SAVE_DIR, exist_ok=True)

weights = [1/6, 1/12] # direct/indirect nb weight

class noisy_spatial_imputer:
    def __init__(self, mask, dataset, noise = 0.01):
        """
        weight: list(direct, indirect nb weight)
        mask: list of channel id to be imputed
        dataset: dataset hard-coded info
        """
        self.dataset = dataset
        self.noise = noise
        self.imputed_id = mask
        self.n_imputed = len(mask)

        self.valid = np.ones(len(mask)) # valid: of all to be imputed, if any is connected in the system 
        for i, t in enumerate(self.imputed_id):
            neighbors = np.array(list(self.dataset.dn_id[t]) + list(self.dataset.idn_id[t]))+t
            if np.any(np.isin(neighbors, self.imputed_id)): 
                self.valid[i] = 0
    

        # print(self.n_imputed, self.valid, self.imputed_id)
    def _neighbor(self, idx):
        dnw = 4/len(self.dataset.dn_id[idx])*weights[0]
        idnw = 4/len(self.dataset.idn_id[idx])*weights[1]
        return [tuple([dnw, dn+idx]) for dn in self.dataset.dn_id[idx]] + [tuple([idnw, idn+idx]) for idn in self.dataset.idn_id[idx]]

    def _construct_eqsys(self,trial):
        coords_to_vidx= np.zeros(trial.shape[0]) 
        coords_to_vidx[self.imputed_id] = np.arange(self.n_imputed)
        coords_to_vidx = coords_to_vidx.astype(np.int32)

        # x(1, 562) = sum xother(1, 562)*w
        A = lil_matrix((self.n_imputed, self.n_imputed)) # (rid, cid) , val / each eq is a combination of other eqs / cool: weights
        b = np.zeros((self.n_imputed, trial.shape[1])) # zero vector
        sum_neighbors = np.ones(self.n_imputed) # sum of weights is 1

        for i, target in enumerate(self.imputed_id): # 0, 7/c5, dn = 1, idn = -5,7 # [(1/6, 1), (1/12, -5), [(1/12, 7)]]
            neighbor = self._neighbor(target)
            # print(neighbor)
            for n in neighbor:
                offset, weight = n[1], n[0]
                
                # if self.valid[i] == 1: # can be constructed using some chs
                b[i, :] -= weight * trial[offset, :]
            if not self.valid[i]:
                A[i, coords_to_vidx[i]] = weight
                sum_neighbors[i] = sum_neighbors[i] - weight
            
        A[np.arange(self.n_imputed),np.arange(self.n_imputed)] = -sum_neighbors

        return A, b


    def _return_imputed(self, trial):
        perturbed = trial.copy()
        A, b = self._construct_eqsys(trial)
        res = torch.tensor(spsolve(csc_matrix(A), b))
        perturbed[self.imputed_id, :] = res + np.random.randn(*res.shape) * self.noise
        # perturbed -= perturbed.mean() - trial.mean()
        perturbed = (perturbed - perturbed.min())/(perturbed.max()-perturbed.min())
        perturbed = perturbed * (trial.max()-trial.min()) + trial.min()
        return perturbed

class dataset:
    def __init__(self, n_ch, n_tsamp):
        self.n_ch=n_ch
        self.n_tsamp = n_tsamp
        self.ch_name = None
        self.dn_id = None
        self.ind_id = None

class MI(dataset):
    def __init__(self, n_ch=22, n_tsamp=562):
        self.ch_id = np.arange(n_ch)
        self.ch_name = [   'Fz', 
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
                    'P1', 'Pz', 'P2', 
                           'POz']
        self.dn_id = [(3,),
            (1,6),(-1, 1, 6), (-1, 1, -3, 6), (-1, 1, 6),(1,6),
            (1,), (-1, 1, -6, 7), (-1, 1, -6, 7),(-1, 1, -6, 7),(-1, 1, -6, 7),(-1, 1, -6, 7), (-1, ),
            (-6, 1), (-1, 1, -6, 4), (-1, 1, -6, 4), (-1, 1, -6, 4), (-1, -6),
            (-4, 1),(-1, 1, -4, 2), (-14, -1),
            (-2, )]
        self.idn_id = [(2, 4),
            (5,7), (5,7), (5,7), (5,7), (5,7),
            (-5,7), (-5,7), (-7, -5, 5, 7), (-7, -5, 5, 7),(-7, -5, 5, 7), (-7, 5), (-7, 5),
            (-7, -5, 5),(-7, -5, 5), (-7, -5, 3, 5), (-7, -5, 3), (-7, -5, 3),
            (-5, -3, 3), (-5, -3), (-5, -3, 1),
            (-3, -1)]

class ERN(dataset):
    def __init__(self, n_ch=56, n_tsamp=161):
        self.ch_id = np.arange(n_ch)
        self.ch_name = ['Fp1',                             'Fp2',
                    'AF7', 'AF3',                      'AF4', 'AF8',
              'F7',  'F5',  'F3',  'F1',  'Fz',  'F2',  'F4',  'F6',  'F8',
             'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
              'T7',  'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',  'T8',
             'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
              'P7',  'P5',  'P3',  'P1',  'Pz',  'P2',  'P4',  'P6',  'P8',
                       'PO7',            'POz',            'PO8',
                            'O1',                         'O2']

        self.dn_id = [(3,4),(3,4), 
            (1,5),(-3,-1,5),(-3, 1, 8),(-1,8),
        (1,9),(-5,-1, 1, 9),(-5,-1, 1, 9),(-1,1,9),(-1,1,9),(-1,1,9),(-8,-1,1,9),(-8,-1,1,9),(-1,9),
        (-9,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,9),
        (-9,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,9),
        (-9,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,1,9),(-9,-1,9),
        (-9,1),(-9,-1,1,8),(-9,-1,1),(-9,-1,1),(-9,-1,1,6),(-9,-1,1),(-9,-1,1),(-9,-1,1,4),(-9,-1),
        (-8,3),(-6,),(-4,2),
        (-3, 1),(-2,-1)
        ]

        self.idn_id = [(1,2),(-1,2),
            (-2,4,6),(4,6),(7,9),(-4,7,9),
        (-4,10),(-4,8,10),(-6,8,10),(-6,8,10),(8,10),(-7,8,10),(-7,8,10),(-9,8,10),(-9,8),
        (-8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,8),
        (-8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,8),
        (-8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,-8,8,10),(-10,8),
        (-8,9),(-10,-8),(-10,-8,7),(-10,-8,7),(-10,-8),(-10,-8,5),(-10,-8,5),(-10,-8),(-10,4),
        (-9,-7),(-7,-5),(-5, -3),
        (-2,),(-3,)
        ]

class SSVEP(dataset):
    def __init__(self, n_ch=8, n_tsamp=100):
        self.ch_id = np.arange(n_ch)
        self.ch_name = ['PO7', 'PO3', 'O1', 'POz', 'Oz', 'PO4', 'O2', 'PO8']
        # 'PO7', 'PO3',  'POz',  'PO4',  'PO8'
        #         'O1',   'Oz',   'O2',
        self.dn_id  = [(1,),(-1,1,2),(-1,2),(-2,2,1),(-2,2,-1),(-2,1,2),(-2,-1),(-2,)]
        self.idn_id = [(2,),(3,),(-2, 1),(-1,3),(-3,1),(-1,),(-3,1),(-1,)]


modes = [('mo',-1), ('le',1)]

ern_subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


def interp_test(model, modelname, benigns, chrank, labels, bn, sub, method):
    benigns = benigns.squeeze()
    mixtures = benigns.copy()

    for t in range(benigns.shape[0]):
        nsi = noisy_spatial_imputer(chrank[t], datastruct, np.std(benigns[t]))
        mixtures[t, chrank[t]] = nsi._return_imputed(benigns[t])[chrank[t]]
    
    _, test_loader_r = get_loader(bn, mixtures, labels)
    loss_fn = nn.CrossEntropyLoss()
    # for MI,, SSVEP
    test_loss, test_acc, test_out = evaluate_an_epoch(model,device, test_loader_r, loss_fn)
    ## for ERN
    # test_loss, test_acc, test_out = evaluate_an_epoch_auc(model,device, test_loader_r, loss_fn)
    
    return test_loss, test_acc, test_out, mixtures

## for MI
kwerg = dict(n_classes=4,
            channels=22,
            samples=562,
            sfreq=125.0)
datastruct = MI()


## for ERN
# kwerg = dict(n_classes=2,
#             channels=56,
#             samples=160,
#             sfreq=128)
# datastruct = ERN()

## for SSVEP
# kwerg = dict(n_classes=5,
#             channels=8,
#             samples=125,
#             sfreq=250,)
# datastruct = SSVEP()

if __name__ == '__main__':
    ## for MI, ERN
    model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    ## for SSVEP
    # model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}

    leg = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random']
    for m in model_type.keys():
        ## run single model
        # if m != 'eegnet':
        #     continue
        filler = '' if args.absbool ==1 else 'n'
        hists, hists1 = [], []

        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_road/{dataname}')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_road/{dataname}'))

        for l in leg:
            if args.absbool ==1 and l in ['smoothgrad_sq', 'vargrad']:
                continue
            testmodel = model_type[m](**kwerg)

            print(m, l)
            # adjust according to dataset
            dim = 22
            ## for MI
            hist  = dict(acc=np.zeros((9, dim)), loss=np.zeros((9,dim)), out=np.zeros((9, dim, 288, 4)))
            hist1 = dict(acc=np.zeros((9, dim)), loss=np.zeros((9,dim)), out=np.zeros((9, dim, 288, 4)))
            ## for ERN
            # hist =  dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
            # hist1 = dict(acc=np.zeros((16, dim)), loss=np.zeros((16, dim)), out=np.zeros((16, dim, 40, 2)))
            ## for SSVEP
            # hist =  dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
            # hist1 = dict(acc=np.zeros((11, dim)), loss=np.zeros((11, dim)), out=np.zeros((11, dim, 100, 5)))
            for s in tqdm(range(11)): #tqdm(range(len(ern_subs))): ## change accordingly
                
                ## for ERN
                # sub = ern_subs[s]
                ## for MI, SSVEP
                sub = s+1

                model_path = os.path.join(SAVE_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname, rep, sub, m))
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                testmodel.load_state_dict(checkpoint["state_dict"]) 
                testmodel = testmodel.to(device)
                
                ## for MI
                mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
                ## for ERN
                # mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
                ## for SSVEP
                # mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
                xtest, ytest = mat["x_test"], mat["y_test"].squeeze()
                if l != 'gradientxinput':
                    grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), allow_pickle=True)
                else:
                    grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'), allow_pickle=True)
                    ## for MI, ERN & SSVEP respectively
                    grads = np.multiply(grads, xtest[:])
                    # grads = np.multiply(grads, xtest[-40:])
                    # grads = np.multiply(grads, xtest[-100:])
                
                if args.absbool:
                    grads = np.absolute(grads)

                for k in range(1, xtest.shape[1]+1):
                    chrankM = np.argsort(grads.sum(axis=-1)*modes[0][1], axis=-1)[:, :k+1]
                    chrankL = np.argsort(grads.sum(axis=-1)*modes[1][1], axis=-1)[:, :k+1]

                    ## for MI, ERN & SSVEP respectively
                    loss, acc, out, res = interp_test(testmodel,m, xtest, chrankM,  ytest, 32, sub, l)
                    # loss, acc, out = interp_test(testmodel,m, xtest[-40:], chrankM,  ytest[-40:], 25, sub, l)
                    # loss, acc, out = interp_test(testmodel,m, xtest[-100:], chrankM,  ytest[-100:], 25, sub, l)
                    hist['acc'][s][k-1] = acc
                    hist['loss'][s][k-1] = loss
                    hist['out'][s][k-1] = out

                     ## for MI, ERN & SSVEP respectively
                    loss, acc, out, res = interp_test(testmodel,m, xtest, chrankL,  ytest, 32,sub, l)
                    # loss, acc, out = interp_test(testmodel,m, xtest[-40:], chrankL,  ytest[-40:], 25, sub, l)
                    # loss, acc, out = interp_test(testmodel,m, xtest[-100:], chrankL,  ytest[-100:], 25, sub, l)
                    np.save(f'sub{sub}_{k}_chroad_le.npy', res)
                    hist1['acc'][s][k-1] = acc
                    hist1['loss'][s][k-1] = loss
                    hist1['out'][s][k-1] = out
            hists.append(hist)    
            hists1.append(hist1)

        filler = '' if args.absbool ==1 else 'n'

        with open(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_road/{dataname}/{m}_ch_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as handle:
            pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
        with open(os.path.join(SAVE_DIR, f'repeat{rep}/ch_test_road/{dataname}/{m}_ch_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as handle:
            pickle.dump(hists1, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()