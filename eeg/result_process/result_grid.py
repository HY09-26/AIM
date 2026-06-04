import pickle
import numpy as np
import os
import math
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("fid", type=int)
args = parser.parse_args()


#================================
# plot ranking grid for methods (supplementary), dataset-domain-model-repeat averaged
#================================ 

# leg = ['gradient', 'gradInput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'integrated_grad', 'random', 'gradient(abs)','gradInput(abs)', 'smoothgrad(abs)', 'integrated_grad(abs)']
leg = ['GD', 'GI', 'SG', 'SS', 'VG', 'IG', 'GDA','GIA', 'SGA', 'IGA', 'RD']
models=['eegnet', 'icnn', 'sccnet']
datasets = ['SMR', 'ERN', 'SSVEP']
frames = ['road', 'ae']
frames_titles = ['ROAD', 'AR']
domains = ['ch', 'ts', 'fq']
metrics = ['AOC', 'ABC', 'AUC']


clr = sns.color_palette("coolwarm", len(leg)).as_hex()

frame = frames[args.fid]
frame_title = frames_titles[args.fid]

if __name__ == "__main__":
    base_dir = f"C:/Users/irish/DATA/atk/figures/acc_mat/"

    # =======================================================================================
    aopc = np.load(os.path.join(base_dir, f'AOPC_{frame}.npy'))
    abpc = np.load(os.path.join(base_dir, f'ABPC_{frame}.npy'))
    aupc = np.load(os.path.join(base_dir, f'AUPC_{frame}.npy')) # 3 dataset, 3 domain * (3 model * 5 repeat), 11 method


    # daopc = aopc[1].squeeze()
    daopc = np.reshape(aopc, (3, 3, 15, len(leg))).mean(axis=2) # 3 dataset, 3 domain, 11 method
    dabpc = np.reshape(abpc, (3, 3, 15, len(leg))).mean(axis=2)
    daupc = np.reshape(aupc, (3, 3, 15, len(leg))).mean(axis=2)

    
    for t in range(3):
        for d in range(3):
            daopc[t, d] = rankdata(daopc[t, d]*-1)
            dabpc[t, d] = rankdata(dabpc[t, d]*-1)
            daupc[t, d] = rankdata(daupc[t, d]*-1)
        

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize = (15,5))
    plt.subplots_adjust(hspace=0.05)
    plt.subplots_adjust(wspace=0.3)
    cmap = 'copper_r'

    for i in range(3): # metric
        for j in range(3): # dataset
            ax[i, j].set_xticks(np.arange(len(leg)*3), leg * 3,  fontsize=8)
            if j ==0:
                ax[i, j].set_ylabel(metrics[i])
            ax[i, j].set_yticks(np.arange(3), ['Spatial', 'Temporal', 'Spectral'])
            if i ==0:
                ax[i, j].set_xlabel(datasets[j])
                ax[i, j].xaxis.set_label_position('top')
                ax[i, j].imshow(daopc[j], cmap = cmap)
            elif i ==1:
                ax[i, j].imshow(dabpc[j], cmap = cmap)
            elif i ==2:
                im = ax[i, j].imshow(daupc[j], cmap = cmap)
    for j in range(3): # dataset
        for k in range(3): # domain
            for l in range(len(leg)):
                text = ax[0,j].text(l, k, int(daopc[j, k,l]),
                            ha="center", va="center", color="k")
                text = ax[1,j].text(l, k, int(dabpc[j, k,l]),
                            ha="center", va="center", color="k")
                text = ax[2,j].text(l, k, int(daupc[j, k,l]),
                            ha="center", va="center", color="k")
    

    fig.suptitle(f'md{frame_title} rankings', x = 0.435, y= 0.93)
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()