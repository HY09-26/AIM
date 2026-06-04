import pickle
import numpy as np
import os
import math
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("fid", type=int)
parser.add_argument("did", type=int)
# parser.add_argument("mid", type=int)
args = parser.parse_args()


def minmax(arr, Max=0, Min=0):
    Min = arr.min() if (Min==0 and Max ==0) else Min
    Max = arr.max() if Max==0 else Max
    arr = np.clip(arr, Min, Max)
    if len(arr.shape)>1:
        for i in range(arr.shape[0]):
            arr[i] =  (arr[i]-Min)/(Max-Min)
    else:
        arr =  (arr-Min)/(Max-Min)
    return arr

   

#================================
# compute area-metrics for single repeat (supplementary)
#================================ 
  


leg = ['GD', 'GI', 'SG', 'SS', 'VG', 'IG', 
       'GDA','GIA', 'SGA',           'IGA', 'RD']
models=['eegnet', 'icnn', 'sccnet']
frames = ['road', 'ae']
domains = ['ch', 'ts', 'fq']
domains_titles = ['Spatial', 'Temporal', 'Spectral', ]

clr = ['#e31a1c', '#ff7f00', '#33a02c', '#1f78b4', '#6a3d9a', '#ec4bf3',\
       '#e31a1c', '#ff7f00', '#33a02c', '#ec4bf3', 'k'] 

# dataset = 'MI'
# chance = 0.25

# dataset = 'ERN'
# chance = 0.5

dataset = 'SSVEP'
chance = 0.2

frame = frames[args.fid]
domain = domains[args.did]
domain_title = domains_titles[args.did]

if __name__ == "__main__":
    base_dir = f"C:/Users/irish/DATA/atk/figures/acc_mat/{frame}/{dataset}/{domain}"

    # =======================================================================================
    aopc, abpc, aupc = None, None, None
    print(dataset, frame, domain)
    for mid in range(3):
        model = models[mid]

        dim = 0
        if aopc is None: # repeat, model, method
            aopc, abpc, aupc = np.zeros((5, 3, len(leg))), np.zeros((5, 3, len(leg))), np.zeros((5, 3, len(leg)))

        filler = '' # extended experiment: filler string in filename
        for rep in range(5):
            mM = loadmat(os.path.join(base_dir, f'{model}_{domain}_morf_rep{rep}{filler}.mat'))
            mL = loadmat(os.path.join(base_dir, f'{model}_{domain}_lerf_rep{rep}{filler}.mat'))

            if dim == 0:
                if dataset != 'ERN' and domain=='ch':
                    dim = mM[leg[0]].squeeze().shape[0]//2
                else:
                    dim = mM[leg[0]].squeeze().shape[0]-1
                if mid ==0:
                    print(dim)

            for i in range(len(leg)):
                curveM = np.clip(mM[leg[i]].squeeze()[1:dim+1], chance, mM[leg[0]].squeeze()[0])
                curveL = np.clip(mL[leg[i]].squeeze()[1:dim+1], chance, mM[leg[0]].squeeze()[0])

                ao = 1 - curveM
                ab = np.clip(curveL-curveM,  a_min = 0, a_max=None)
                au = curveL #- chance
                
                aopc[rep, mid, i] += ao.mean()
                abpc[rep, mid, i] += ab.mean()
                aupc[rep, mid, i] += au.mean()

    aopc = np.reshape(aopc, (15,len(leg))) # eliminate repeat & model dimension
    abpc = np.reshape(abpc, (15,len(leg)))
    aupc = np.reshape(aupc, (15,len(leg)))



    tablao, tablab, tablau = [], [], []
    tablao += [str(np.round(mu, 3)).ljust(5, '0')+'$\pm$'+str(np.round(sig, 3)).ljust(5, '0') for mu, sig in zip(aopc.mean(axis=0), aopc.std(axis=0))]
    tablab += [str(np.round(mu, 3)).ljust(5, '0')+'$\pm$'+str(np.round(sig, 3)).ljust(5, '0') for mu, sig in zip(abpc.mean(axis=0), abpc.std(axis=0))]
    tablau += [str(np.round(mu, 3)).ljust(5, '0')+'$\pm$'+str(np.round(sig, 3)).ljust(5, '0') for mu, sig in zip(aupc.mean(axis=0), aupc.std(axis=0))]
    for i in range(len(leg)):
        tabl = [tablao[i], tablab[i], tablau[i]]
        print('\\textbf{'+ leg[i] + '} &', ' & '.join(tabl)) #, '\\\\')

          

