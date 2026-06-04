import numpy as np
import os
import math
import random
import pickle
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("fid", type=int)
parser.add_argument("did", type=int)
parser.add_argument("mid", type=int)
args = parser.parse_args()


clr = ['#e31a1c', '#ff7f00', '#33a02c', '#1f78b4', '#6a3d9a', '#fb9a99',\
      'k',
       '#e31a1c', '#ff7f00', '#33a02c', '#fb9a99']
leg = ['GD', 'GI', 'SG', 'SS', 'VG', 'IG', 
       'RD', 
       'GDA','GIA', 'SGA', 'IGA']


#================================
# 5 repeats
# 2 framework
# 3 dataset
# 3 domains
# 3 models


# plot performance-ratio curve of a single repeat (for supplemantary)
# save dataset-frame-model-domain-repeat subject averaged acc results
#================================



def plot_hist(dir, maccs, hist1, hist2, model = 'eegnet', title='', leng=100, beg=0, idc=None,  dom='ch'):
    fig = plt.figure(figsize=(8,6))
    
    nsub = hist1[0]['acc'].shape[0]

    percent = 1 if (args.did==0) else 5
    hist_accs = dict()

    for i in range(len(hist1)-1):
        accs = np.zeros(leng-beg+1) if idc is None else np.zeros(len(idc)+1)

        for sub in range(nsub):
            accs[0] += maccs[sub]
            if idc is None:
                accs[1:] += hist1[i]['acc'][sub][beg:leng]
            else:
                accs[1:] += hist1[i]['acc'][sub][idc]
        if idc is None:
            plt.plot(np.arange(0,leng-beg+1)*percent, accs[:]/nsub, alpha=0.6, ls='-', label=leg[i], c=clr[i])
        else:
            plt.plot(np.arange(len(idc)+1)*percent, accs[:]/nsub, alpha=0.6, ls='-', label=leg[i], c=clr[i])
        hist_accs[leg[i]] =  accs[:]/nsub
    
    
    for i in range(len(hist2)):
        accs = np.zeros(leng-beg+1) if idc is None else np.zeros(len(idc)+1)
        for sub in range(nsub):
            accs[0] += maccs[sub]
            
            if idc is None:
                accs[1:] += hist2[i]['acc'][sub][beg:leng]
            else:
                accs[1:] += hist2[i]['acc'][sub][idc]
            
        if idc is None:
            plt.plot(np.arange(0,leng-beg+1)*percent, accs[:]/nsub, alpha=0.6, ls='--', label=leg[i+len(hist1)], c=clr[i+len(hist1)])
        else:
            plt.plot(np.arange(len(idc)+1)*percent, accs[:]/nsub, alpha=0.6, ls='--', label=leg[i+len(hist1)], c=clr[i+len(hist1)])
        hist_accs[leg[i+len(hist1)]] =  accs[:]/nsub

    # random 
    accs = np.zeros(leng-beg+1) if idc is None else np.zeros(len(idc)+1)
    for sub in range(nsub):
        accs[0] += maccs[sub]
        if idc is None:
            accs[1:] += hist1[-1]['acc'][sub][beg:leng]
        else:
            accs[1:] += hist1[-1]['acc'][sub][idc]
    if idc is None:
        plt.plot(np.arange(0,leng-beg+1)*percent, accs[:]/nsub, alpha=0.6, ls='dashdot', label=leg[len(hist1)-1], c='k')
    else:
        plt.plot(np.arange(len(idc)+1)*percent, accs[:]/nsub, alpha=0.6, ls='dashdot', label=leg[len(hist1)-1], c='k')
 
    hist_accs[leg[len(hist1)-1]] =  accs[:]/nsub
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(leg))
    plt.ylabel('Accuracy')
    # plt.xticks(np.arange( leng//5+1)*5, np.arange(leng//5+1, dtype=np.int32)*5)
    # if args.did == 0:
    #     plt.xlabel("#Channels Imputed")
    # elif args.did ==1:
    #      plt.xlabel("%Time samples Imputed")
    # elif args.did ==2:
    #     plt.xlabel("%Frequencies Imputed")
    
    plt.title(model[:4].upper()+model[4:]+title)
    # plt.savefig(os.path.join(dir, f'{dom}_{model}'+title.lower()[:5]+f'_rep{title[-2]}_si'))
    # plt.show()
    # plt.close()

    # os.makedirs(f'figures/acc_mat/{frame}/{dataset}/{dom}', exist_ok=True)
    savemat(f'figures/acc_mat/{frame}/{dataset}/{dom}/{model}_{dom}{title.lower()[:5]}_rep{title[-2]}_full.mat', hist_accs)


models=['eegnet', 'icnn', 'sccnet']
frames=['road', 'ae']
domains = ['ch', 'ts', 'fq']


dim = 10
dataset = 'SSVEP'
mid = args.mid
model = models[args.mid]
domain =domains[args.did]
frame = frames[args.fid]




if __name__ == "__main__":
    
    accs = np.load(f'model_accs_{dataset}.npy')
    print(dataset, frame, domain, model)

    for rep in range(5):
    
        base_dir = f"C:/Users/irish/DATA/atk/repeat{rep}/{domain}_test_{frame}_val/{dataset}" 
        if domain == 'fq' and frame =='ae':
            base_dir += '/fft_full_single'
            # base_dir += '/fft_half_single'
        save_dir = f"C:/Users/irish/DATA/atk/figures/{frame}/{domain}/{dataset}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        histsm, histsl = None, None # 5* hist{acc(9,22), loss(9,22), out(9,22,288,4)}
        filler = '' # filler string in filename for extended experiments
        with open(os.path.join(base_dir,  f'{model}_{domain}_nabs_morf{filler}.pickle'), 'rb') as handle:
            histsm = pickle.load(handle)
            handle.close()
        with open(os.path.join(base_dir,  f'{model}_{domain}_nabs_lerf{filler}.pickle'), 'rb') as handle:
            histsl = pickle.load(handle)
            handle.close()

        histsm0, histsl0 = None, None
        with open(os.path.join(base_dir,  f'{model}_{domain}_abs_morf{filler}.pickle'), 'rb') as handle:
            histsm0 = pickle.load(handle)
            handle.close()
        with open(os.path.join(base_dir,  f'{model}_{domain}_abs_lerf{filler}.pickle'), 'rb') as handle:
            histsl0 = pickle.load(handle)
            handle.close()
        
        
        if rep ==0:
            print(histsm[-1]['acc'].shape, histsl[-1]['acc'].shape, histsm0[-1]['acc'].shape, histsl0[-1]['acc'].shape)
        idc = np.arange(1, 11)*5
    
        if domain != 'ch' and histsm[-1]['acc'].shape[-1] >11 and False:
            histsm[-1]['acc'][:, idc] = (histsm[-1]['acc'][:, idc] + histsm0[-1]['acc'][:, idc])/2
            histsl[-1]['acc'][:, idc] = (histsl[-1]['acc'][:, idc] + histsl0[-1]['acc'][:, idc])/2

            plot_hist(save_dir, accs[rep, mid], histsm, histsm0[:4], model, f'_MoRF (Repeat {rep})', 50, 0, idc, dom=domain)
            plot_hist(save_dir, accs[rep, mid], histsl, histsl0[:4], model, f'_LeRF (Repeat {rep})', 50, 0, idc, dom=domain)
        else:
            histsm[-1]['acc'] = (histsm[-1]['acc'] + histsm0[-1]['acc'])/2
            histsl[-1]['acc'] = (histsl[-1]['acc'] + histsl0[-1]['acc'])/2

            plot_hist(save_dir, accs[rep, mid], histsm, histsm0[:4], model, f'_MoRF (Repeat {rep})', dim, 0, None, dom=domain)
            plot_hist(save_dir, accs[rep, mid], histsl, histsl0[:4], model, f'_LeRF (Repeat {rep})', dim, 0, None, dom=domain)
        
    
    