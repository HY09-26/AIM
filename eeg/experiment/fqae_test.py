import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import train_an_epoch, evaluate_an_epoch, get_loader, getloader, evaluate_an_epoch_auc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--abs_saliency", type=int, choices=[0, 1], required=True, help="1 = absolute saliency")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# np.set_printoptions(precision=8) 
# torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

torch.use_deterministic_algorithms(True)

dataname = 'ERN'
DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/P300"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"

modes = [('mo',-1), ('le',1)]


ern_subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


def find_neighbors(den, grad, ratio, absbool, mode):
    # den shape: ch, fbin
    # ratio: 0~1 float
    # print(den.shape, grad.shape)
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

def freq_interp_test(model, grad, benign, ae, bn, labels, sub=1, absbool=1,  mode = modes[0], sfreq = 125): # 22, 562
    (tr, ch, ts) =  benign.shape

    if absbool == 1:
        grad = np.absolute(grad, out=grad)
        
    freqs = np.fft.fftfreq(ts, d = 1/sfreq) 
    uq_freqs = np.where(np.isclose(freqs, -1*(sfreq/2), 0.5))[0][0] #281, 0~-62.5
    
    ## removing DC offset
    benign_dc, ae_dc = benign.mean(axis=-1)[:,:, None], ae.mean(axis=-1)[:,:, None]
    benign -= np.repeat(benign_dc, ts, axis = -1)
    ae -= np.repeat(ae_dc, ts, axis = -1)

    fseries_grad = np.fft.fft(grad)
    fseries_data = np.fft.fft(benign)
    fseries_mixture = fseries_data.copy()
    fseries_ae = np.fft.fft(ae)


    accs, losses, outs = np.zeros(100), np.zeros(100), np.zeros((100, tr, 2))
    
    
    for i in range(1, 11):
        for t in range(tr):
            fs_grad = fseries_grad[t]
            fs_ae = fseries_ae[t]
            fs_data = fseries_data[t]
            fs_data2 = fs_data.copy()

            cid, neighbor_tup, tt = find_neighbors(fs_data[:, 1:uq_freqs], abs(fs_grad[:, 1:uq_freqs]), np.round(i*0.05, 2), absbool, mode[0])

            fseries_mixture[t, :, neighbor_tup[0]:neighbor_tup[1]+1] = fs_ae[:, neighbor_tup[0]:neighbor_tup[1]+1]           
            fseries_mixture[t, :, ts-neighbor_tup[1]:ts-neighbor_tup[0]+1] = fs_ae[:, ts-neighbor_tup[1]:ts-neighbor_tup[0]+1]

        
        _, test_loader_r = get_loader(bn, np.fft.ifft(fseries_mixture).real, labels)
        loss_fn = nn.CrossEntropyLoss()
        # losses[i-1], accs[i-1], outs[i-1] = evaluate_an_epoch(model, device, test_loader_r, loss_fn)
        losses[i-1], accs[i-1], outs[i-1] = evaluate_an_epoch_auc(model, device, test_loader_r, loss_fn)
    return losses, accs, outs

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
    leg = ['gradient','gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad','random']
    filler = '' if args.abs_saliency ==1 else 'n'
    
    
    for rep in range(5):
        AE_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/ae/{dataname}"
        EXPL_DIR = f"/mnt/left/home/2023/irishsieh/atk/repeat{rep}/expl/{dataname}"

        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_ae/{dataname}')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_ae/{dataname}'))
        if not os.path.exists(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_ae/{dataname}/fft_full_single')):
            os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_ae/{dataname}/fft_full_single'))

        for m in model_type.keys():
            # if m != 'eegnet':
            #     continue

            hists, hists1 = [], []

            
            for l in leg[:]:
                
                if args.abs_saliency ==1 and l in ['smoothgrad_sq', 'vargrad',]:
                    continue

                testmodel = model_type[m](**kwerg)

                print('rep', rep, m, l, 'mode', args.abs_saliency)
                # hist = dict(acc=np.zeros((9, 100)), loss=np.zeros((9,100)), out=np.zeros((9, 100, 288, 4)))
                # hist1 = dict(acc=np.zeros((9, 100)), loss=np.zeros((9,100)), out=np.zeros((9, 100, 288, 4)))
                hist =  dict(acc=np.zeros((16, 100)), loss=np.zeros((16,100)), out=np.zeros((16, 100, 40, 2)))
                hist1 = dict(acc=np.zeros((16, 100)), loss=np.zeros((16,100)), out=np.zeros((16, 100, 40, 2)))
                # hist =  dict(acc=np.zeros((11, 100)), loss=np.zeros((11,100)), out=np.zeros((11, 100, 100, 5)))
                # hist1 = dict(acc=np.zeros((11, 100)), loss=np.zeros((11,100)), out=np.zeros((11, 100, 100, 5)))


                for s in tqdm(range(len(ern_subs))):
                    
                    sub = ern_subs[s]
                    # sub = s+1

                    model_path = os.path.join(SAVE_DIR, "models/{}/bests_repeat{}/sub{}-{}.pth".format(dataname, rep, sub, m))
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    testmodel.load_state_dict(checkpoint["state_dict"]) 
                    testmodel = testmodel.to(device)
                    
                    
                    # mat = loadmat(os.path.join(DATASET_DIR, f"BCIC_S{sub:02d}_E.mat"))
                    mat = loadmat(os.path.join(DATASET_DIR, f"Data_S{sub:02d}_Sess.mat"))
                    # mat = loadmat(os.path.join(DATASET_DIR, f"U0{sub:02d}.mat"))
                    xtest, ytest = mat["x_test"], mat["y_test"].squeeze()
                    if l != 'gradientxinput':
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'), allow_pickle=True)
                    else:
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'), allow_pickle=True)
                        grads = np.multiply(grads, xtest[-40:])
                    
                    ae = np.load(os.path.join(AE_DIR, f'{m}/sub{sub}.npy')).squeeze()
                    
                    # loss, acc, out = freq_interp_test(testmodel, grads, xtest, ae, 32, ytest, sub, args.abs_saliency, mode = modes[0], sfreq=125)
                    loss, acc, out = freq_interp_test(testmodel, grads, xtest[-40:], ae, 32, ytest[-40:], sub, args.abs_saliency, mode = modes[0], sfreq=128)
                    # loss, acc, out = freq_interp_test(testmodel, grads, xtest[-100:], ae, 25, ytest[-100:], sub, args.abs_saliency, mode = modes[0], sfreq=125)
                    hist['acc' ][s] = acc
                    hist['loss'][s] = loss
                    hist['out' ][s] = out

                    # loss, acc, out = freq_interp_test(testmodel, grads, xtest, ae, 32, ytest, sub, args.abs_saliency, mode = modes[1], sfreq=125)
                    loss, acc, out = freq_interp_test(testmodel, grads, xtest[-40:], ae, 32, ytest[-40:], sub, args.abs_saliency, mode = modes[1], sfreq=128)
                    # loss, acc, out = freq_interp_test(testmodel, grads, xtest[-100:], ae, 25, ytest[-100:], sub, args.abs_saliency, mode = modes[1], sfreq=125)
                    hist1['acc' ][s] = acc
                    hist1['loss'][s] = loss
                    hist1['out' ][s] = out
                hists.append(hist) 
                hists1.append(hist1)

             

            with open(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_ae/{dataname}/fft_full_single/{m}_fq_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
            with open(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_ae/{dataname}/fft_full_single/{m}_fq_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as handle:
                pickle.dump(hists1, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()