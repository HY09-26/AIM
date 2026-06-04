import numpy as np
import os
import math
import random
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.stats import rankdata


# ch_names = ['Fz', 
#             'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
#             'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
#             'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
#             'P1', 'Pz', 'P2', 
#             'POz']
# label = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']


# ch_names = ['Fp1',                             'Fp2',
#     'AF7', 'AF3',                      'AF4', 'AF8',
# 'F7',  'F5',  'F3',  'F1',  'Fz',  'F2',  'F4',  'F6',  'F8',
# 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
# 'T7',  'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',  'T8',
# 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
# 'P7',  'P5',  'P3',  'P1',  'Pz',  'P2',  'P4',  'P6',  'P8',
#         'PO7',            'POz',            'PO8',
#             'O1',                         'O2']
# label = ['no ERN', 'with ERN']

ch_names = ['PO7', 'PO3', 'O1', 'POz', 'Oz', 'PO4', 'O2', 'PO8']
label = ['12Hz', '10Hz', '8.57Hz', '7.5Hz', '6.6Hz']


def minmax(arr, Max=0, Min=0):
    # if Max <= Min:
    #     arr = (arr-arr.min())/(arr.max()-arr.min() +1e-10)
    # else:
    #     arr =  (arr-Min)/(Max-Min)
    # return arr
    if len(arr.shape) >1:
        for t in range(arr.shape[0]):
            arr[t] = (arr[t]-arr[t].min())/(arr[t].max()-arr[t].min())
        return arr
    else:
        return (arr-arr.min())/(arr.max()-arr.min())


def channel_time(grads, y, fn=None, show=1):
    nrows, ncols = 2,3 # len(np.unique(label))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,6))
    ax[-1, -1].axis('off')

    # plt.subplots_adjust(hspace=0.05)
    # plt.subplots_adjust(wspace=0.25)
    for i in range(len(np.unique(y))):
        grad_data = grads[np.where(y==i)].mean(axis=0)
        grad_data = minmax(grad_data)
        # grad_data = (grad_data-grad_data.min())/(grad_data.max()-grad_data.min())
        # timetick = np.array([0, 0.25, 0.5, 0.75, 1, 1.25])
        # timetick = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        timetick = np.array([0, 0.25, 0.5, 0.75, 1])
        try:
            im = ax[i//ncols, i%ncols].imshow(grad_data, interpolation='gaussian', aspect=15, cmap='coolwarm')
            ax[i//ncols, i%ncols].set_title(label[i], fontsize=18)

            ax[i//ncols, i%ncols].set_xticks(ticks = timetick*125, labels=timetick)
            ax[i//ncols, i%ncols].set_yticks(ticks = np.arange(len(ch_names)), labels=ch_names, fontsize=6)
            ax[i//ncols, i%ncols].set_title(label[i], fontsize=18)
        except:
            im = ax[i].imshow(grad_data, interpolation='gaussian', aspect=15, cmap='coolwarm')
            ax[i].set_title(label[i], fontsize=18)

            ax[i].set_xticks(ticks = timetick*125, labels=timetick)
            ax[i].set_yticks(ticks = np.arange(len(ch_names)), labels=ch_names, fontsize=6)
            ax[i].set_title(label[i], fontsize=18)
    fig.supxlabel('Time', fontsize=14)
    fig.supylabel('Channel', fontsize=14)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink = 0.7)
    if show:
        plt.show()
    plt.savefig(fn)

def freq_time(grads, y,  sfreq = 125, fn=None, show=1):
    nrows, ncols = 1, 5 
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (12,6))
    plt.subplots_adjust(wspace=0.4)
    for i in range(len(np.unique(y))):
        
        freqs, timestamps, saliency = signal.stft(
            grads[np.where(y==i)[0]].squeeze(),
            fs=sfreq,
            axis=-1,
            nperseg=sfreq,
            noverlap=sfreq // 2
        )
        # print(freqs)
        beg, end = 4, 51
        print(saliency.shape) # tr, ch, fq, time
        timetick = np.array([0, 0.5, 1])
        # timetick = np.array([0, 0.417, 0.833, 1.25])
        # timetick = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        grad_data = abs(saliency).mean(axis=0).mean(axis = 0)[beg:end, :]#.mean(axis=0)
        try:
            im = ax[i].imshow(grad_data, interpolation='gaussian', aspect = 0.13)

            ax[i].set_yticks(ticks = freqs[beg:end:5]-freqs[beg], labels=freqs[beg:end:5])
            ax[i].set_xticks(ticks = np.arange(len(timetick)), labels=timetick)
            ax[i].set_title(label[i], fontsize = 18)        
        except:
            im = ax[i//ncols, i%ncols].imshow(grad_data, interpolation='gaussian', aspect = 0.13)

            ax[i//ncols, i%ncols].set_yticks(ticks = freqs[beg:end:5]-freqs[beg], labels=freqs[beg:end:5])
            ax[i//ncols, i%ncols].set_xticks(ticks = np.arange(len(timetick)), labels=timetick)
            ax[i//ncols, i%ncols].set_title(label[i], fontsize = 18)   
    fig.supxlabel('Time', fontsize=14)
    fig.supylabel('Frequency', fontsize=14)
        
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink = 0.5)
    if show:
        plt.show()
    plt.savefig(fn)

def topo(grads, y, fn = None, y_pred=None, show=1):
    # grads = minmax(grads)
    montage = mne.channels.make_standard_montage('standard_1020')
    positions = np.array([montage.get_positions()['ch_pos'][ch] for ch in ch_names])
    # corrects = y[np.where(y==y_pred)[0]]
    # print(corrects, corrects.shape, corrects.shape[0]/y.shape[0])

    kwargs = {'pos': positions[:, 0:2],
                        'ch_type': 'eeg',
                        'sensors': True,
                        # 'names': ch_names,
                        'show': False,
                        'extrapolate': 'local',
                        'outlines': 'head',
                        'sphere': (0.0, -0.02, 0.0, 0.12),
                        }
    # nrows, ncols = 2, np.ceil(len(np.unique(label))/2).astype(np.int32)
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6), sharex=True, sharey=True, dpi =120)
    # ax[-1, -1].axis('off')

    plt.subplots_adjust(hspace=0.05)
    plt.subplots_adjust(wspace=0.05)

    plotter = y #corrects


    for i in range(len(np.unique(y))):
       
        grad_data = grads[np.where(plotter==i)[0]].mean(axis=-1).mean(axis=0)
        grad_data = minmax(grad_data)
        # print(np.where(plotter==i)[0].shape, i, grad_data.max(), grad_data.min())
        try:
            im, _ = mne.viz.plot_topomap(data=grad_data, cmap="coolwarm", axes=ax[i//ncols, i%ncols], **kwargs)
            ax[i//ncols, i%ncols].set_title(label[i], fontsize=18)
        except:
            im, _ = mne.viz.plot_topomap(data=grad_data, cmap="coolwarm", axes=ax[i], **kwargs)
            ax[i].set_title(label[i], fontsize=18)
        # ax[i].set_xlabel('Time')
        # ax[i].set_ylabel('Freq')

        # print(grad_data.min(), grad_data.max(), grads[np.where(y==i)[0]].min(),  grads[np.where(y==i)[0]].max())
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink = 0.5)
    
    # fig.suptitle(title)
    # print(plt.gcf().get_size_inches())

    if show:
        plt.show()
    plt.savefig(fn)

if __name__ == "__main__":
    
    rep = 2
    sub = 2
    model = 'eegnet'
    # model = 'icnn'
    method = 'gradient'
    # method = 'smoothgrad'
    # method = 'smoothgrad_sq'
    # method = 'vargrad'
    # method = 'inte_grad'
    expl_dir = f"C:/Users/irish/DATA/atk/expl/SSVEP/"#{model}/""
    # expl_dir = f"C:/Users/irish/DATA/atk/repeat{rep}/"
    data_dir = "C:/Users/irish/DATA/datasets/MAMEM"

    expl = np.load(expl_dir+f'sub{sub}_{method}.npy')
    # y_pred = np.load(f'C:/Users/irish/DATA/atk/repeat{rep}/rep{rep}_{model}_sub{sub}_output.npy')

    # print(np.fft.fftfreq(125, d = 1/125)[:51] )

    # mat = loadmat(os.path.join(data_dir, f'BCIC_S0{sub}_E.mat'))
    # mat = loadmat(os.path.join(data_dir, f"Data_S{sub:02d}_Sess.mat"))
    mat = loadmat(os.path.join(data_dir, f"U0{sub:02d}.mat"))
    xs, ys = mat['x_test'].squeeze()[-100:], mat['y_test'].squeeze()[-100:]

    # expl = np.multiply(expl, xs)
    # expl = abs(expl)
    # topo(expl, ys, fn =expl_dir+f"sub{sub}_{method}", show=0)
    # topo(expl, ys, fn =expl_dir+f"sub{sub}_{method}_abs", show=0)
    # channel_time(expl, ys, fn =expl_dir+f"sub{sub}_{method}", show=0)
    # channel_time(expl, ys, fn =expl_dir+f"sub{sub}_{method}_abs", show=0)
    freq_time(expl, ys, fn =expl_dir+f"sub{sub}_{method}", show=0)
    # freq_time(expl, ys, fn =expl_dir+f"sub{sub}_{method}_abs", show=0)