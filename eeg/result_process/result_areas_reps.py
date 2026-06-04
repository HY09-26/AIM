import pickle
import numpy as np
import os
import math
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
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
# compute area-metrics for all repeats (manuscript table 5.1)
#================================ 
  


# leg = ['gradient', 'gradInput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'integrated_grad', 'random', 'gradient(abs)','gradInput(abs)', 'smoothgrad(abs)', 'integrated_grad(abs)']
leg = ['GD', 'GI', 'SG', 'SS', 'VG', 'IG', 
       'GDA','GIA', 'SGA',           'IGA', 'RD']
models=['eegnet', 'icnn', 'sccnet']
frames = ['road', 'ae']
domains = ['ch', 'ts', 'fq']
domains_titles = ['Spatial', 'Temporal', 'Spectral', ]

clr = ['#e31a1c', '#ff7f00', '#33a02c', '#1f78b4', '#6a3d9a', '#ec4bf3',\
       '#e31a1c', '#ff7f00', '#33a02c', '#ec4bf3', 'k'] 




if __name__ == "__main__":

    # =======================================================================================
    r = None
    spears = None

    tablao, tablab, tablau = [[],[]], [[],[]], [[],[]]
    for fid, frame in enumerate(frames):
        Daopc, Dabpc, Daupc = np.zeros((3, 3, 15, len(leg))), np.zeros((3, 3, 15, len(leg))), np.zeros((3, 3, 15, len(leg))) # 3 domain, 3 dataset, (3 model * 5 repeat), 11 method

        for tid, (dataset, chance) in enumerate([('MI', .25), ('ERN', .5), ('SSVEP', .2)]):
            for did, domain in enumerate(['ch', 'ts', 'fq']):
                print(dataset, frame, domain)
                aopc, abpc, aupc = None, None, None
                base_dir = f"C:/Users/irish/DATA/atk/figures/acc_mat/{frame}/{dataset}/{domain}"
                domain_title = domains_titles[did]
                
                for mid in range(3):
                    model = models[mid]

                    dim = 0
                    if aopc is None: # rep, model, method
                        aopc, abpc, aupc = np.zeros((5, 3, len(leg))), np.zeros((5, 3, len(leg))), np.zeros((5, 3, len(leg)))

                    for rep in range(0, 5):
                        mM = loadmat(os.path.join(base_dir, f'{model}_{domain}_morf_rep{rep}.mat'))
                        mL = loadmat(os.path.join(base_dir, f'{model}_{domain}_lerf_rep{rep}.mat'))

                        if dim == 0:
                            # else:
                            if dataset != 'ERN' and did==0:
                                dim = mM[leg[0]].squeeze().shape[0]//2
                            else:
                                dim = mM[leg[0]].squeeze().shape[0]-1
                            if mid ==0:
                                print(dim)


                        for i in range(len(leg)):
                            curveM = np.clip(mM[leg[i]].squeeze()[1:dim+1], chance, mM[leg[0]].squeeze()[0])
                            curveL = np.clip(mL[leg[i]].squeeze()[1:dim+1], chance, mM[leg[0]].squeeze()[0])

                            ao = 1- curveM
                            ab = np.clip(curveL-curveM,  a_min = 0, a_max=None)
                            au = curveL 

                            aopc[rep, mid, i] = ao.mean()
                            abpc[rep, mid, i] = ab.mean()
                            aupc[rep, mid, i] = au.mean()
                    Daopc[did, tid] = np.reshape(aopc, (15,len(leg)))
                    Dabpc[did, tid] = np.reshape(abpc, (15,len(leg)))
                    Daupc[did, tid] = np.reshape(aupc, (15,len(leg))) # eliminate domain variability

                    # if tid==0 and did==0: # 5.4 check for specific configuration metrics
                    #     repeat, meth = 0, [6,]
                    #     print([leg[me] for me in meth])
                    #     print(str(np.round(Daopc[tid, did*15:(did)*15+5][repeat, meth], 3)))
                    #     print(str(np.round(Dabpc[tid, did*15:(did)*15+5][repeat, meth], 3)))
                    #     print(str(np.round(Daupc[tid, did*15:(did)*15+5][repeat, meth], 3)))

        Daopc = np.reshape(Daopc, (3, 3*15,len(leg)))
        Dabpc = np.reshape(Dabpc, (3, 3*15,len(leg)))
        Daupc = np.reshape(Daupc, (3, 3*15,len(leg)))


        for domain in range(3):
            tablao[fid] += [str(np.round(mu, 3)).lstrip('0').ljust(4, '0')+'$\pm$'+str(np.round(sig, 3)).lstrip('0').ljust(4, '0')\
                             for mu, sig in zip(Daopc[domain].mean(axis=0), Daopc[domain].std(axis=0))]
            tablab[fid] += [str(np.round(mu, 3)).lstrip('0').ljust(4, '0')+'$\pm$'+str(np.round(sig, 3)).lstrip('0').ljust(4, '0')\
                             for mu, sig in zip(Dabpc[domain].mean(axis=0), Dabpc[domain].std(axis=0))]
            tablau[fid] += [str(np.round(mu, 3)).lstrip('0').ljust(4, '0')+'$\pm$'+str(np.round(sig, 3)).lstrip('0').ljust(4, '0')\
                             for mu, sig in zip(Daupc[domain].mean(axis=0), Daupc[domain].std(axis=0))]

    for i in range(len(leg)):
        tabl = []
        for d in range(3):
            tabl += [tablao[0][d*11+i], tablab[0][d*11+i], tablau[0][d*11+i]]
            tabl += [tablao[1][d*11+i], tablab[1][d*11+i], tablau[1][d*11+i]]
        print('\hline \\textbf{'+ leg[i] + '} &', '& '.join(tabl), '\\\\')


          

