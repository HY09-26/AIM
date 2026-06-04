import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--abs_saliency", type=int, choices=[0, 1], required=True, help="1 = absolute saliency")
args = parser.parse_args()

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import train_an_epoch, evaluate_an_epoch, get_loader, getloader, evaluate_an_epoch_auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 0
np.random.seed(seed); random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# ===== CHANGE ONLY THIS LINE =====
dataname = 'ERN'   # 'MI', 'ERN', or 'SSVEP'
# =================================

# ===== PATH CONFIGURATION =====
MY_SAVE_DIR   = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"
IRISHSIEH_DIR = "/mnt/right/alumni/2023/irishsieh/atk"
# ==============================

if dataname == 'MI':
    DATASET_DIR    = "/mnt/right/alumni/2023/irishsieh/datasets/MI"
    kwerg          = dict(n_classes=4, channels=22, samples=562, sfreq=125.0)
    use_ssvep_model= False
    sub_list       = list(range(1, 10))
    n_trials       = 288
    n_classes_out  = 4
    xtest_sl       = slice(None)
    gradxi_sl      = slice(None)
    use_auc        = False
    batch_size     = 32
    sfreq          = 125          # FFT sampling rate
    lowfq, highfq  = 19, 176      # frequency band index range
    rep_range      = range(5)
    def mat_file(s): return f"BCIC_S{s:02d}_E.mat"

elif dataname == 'ERN':
    DATASET_DIR    = "/mnt/right/alumni/2023/irishsieh/datasets/P300"
    kwerg          = dict(n_classes=2, channels=56, samples=160, sfreq=128)
    use_ssvep_model= False
    sub_list       = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    n_trials       = 40
    n_classes_out  = 2
    xtest_sl       = slice(-40, None)
    gradxi_sl      = slice(-40, None)
    use_auc        = True
    batch_size     = 25
    sfreq          = 128
    lowfq, highfq  = 1, 51
    rep_range      = range(5)
    def mat_file(s): return f"Data_S{s:02d}_Sess.mat"

elif dataname == 'SSVEP':
    DATASET_DIR    = "/mnt/right/alumni/2023/irishsieh/datasets/MAMEM"
    kwerg          = dict(n_classes=5, channels=8, samples=125, sfreq=250)
    use_ssvep_model= True
    sub_list       = list(range(1, 12))
    n_trials       = 100
    n_classes_out  = 5
    xtest_sl       = slice(-100, None)
    gradxi_sl      = slice(-100, None)
    use_auc        = False
    batch_size     = 25
    sfreq          = 125          # FFT uses 125 (not 250)
    lowfq, highfq  = 1, 41
    rep_range      = range(1, 3)  # rep 1, 2 only
    def mat_file(s): return f"U0{s:02d}.mat"

SAVE_DIR = MY_SAVE_DIR
modes    = [('mo', -1), ('le', 1)]
n_subs   = len(sub_list)
fq_dim   = 10   # k in range(1,11), step = k*5% of frequency budget


def find_neighbors(den, grad, ratio, absbool, mode):
    den_avg  = abs(den).mean(axis=0)  if len(den.shape)  > 1 else abs(den)
    grad_avg = abs(grad).mean(axis=0) if len(grad.shape) > 1 else abs(grad)
    den_len  = den_avg.shape[-1]

    target    = den_avg.sum() * ratio
    target_hv = target / 2

    accus = np.ones((den_len, 4)) * -den_len
    accus[:, 0], accus[:, 2] = den_avg, den_avg
    accus[:, 1][den_avg >= target] = 0
    accus[:, 3][den_avg >= target] = 0
    grad_accu = np.zeros((den_len, 2))
    grad_accu[:, 0] = grad_avg

    for i in range(den_len - 1):
        ls = np.logical_and(accus[i+1:, 0] < target_hv, accus[i+1:, 1] < 0)
        accus[i+1:, 0][ls]    += den_avg[:den_len-i-1][ls]
        grad_accu[i+1:, 0][ls]+= grad_avg[:den_len-i-1][ls]
        accus[:, 1][np.logical_and(accus[:, 0] >= target_hv, accus[:, 1] < 0)] = i + 1

        rs = np.logical_and(accus[:den_len-i-1, 2] < target_hv, accus[:den_len-i-1, 3] < 0)
        accus[:den_len-i-1, 2][rs]    += den_avg[i+1:][rs]
        grad_accu[:den_len-i-1, 1][rs]+= grad_avg[i+1:][rs]
        accus[:, 3][np.logical_and(accus[:, 2] >= target_hv, accus[:, 3] < 0)] = i + 1

    valid_lr    = np.logical_and(accus[:, 3] >= 0, accus[:, 1] >= 0)
    neighborhood = np.zeros(den_len)
    neighborhood[valid_lr] = (grad_accu.sum(axis=-1)[valid_lr]
                               / (accus[valid_lr, 1] + accus[valid_lr, 3] + 1))

    inv_l = np.arange(den_len)[(accus[:, 1] < 0)]
    inv_r = np.arange(den_len)[(accus[:, 3] < 0)]

    for il in inv_l:
        ll = 1
        while accus[il, 0] < target and il + ll < den_len:
            accus[il, 0]    = den_avg[:il+ll].sum()
            accus[il, 1]    = ll * -1
            grad_accu[il,0] = grad_avg[:il+ll].sum()
            ll += 1
    for ir in inv_r:
        rr = 1
        while accus[ir, 2] < target and rr <= ir:
            accus[ir, 2]    = den_avg[ir-rr:].sum()
            accus[ir, 3]    = rr * -1
            grad_accu[ir,1] = grad_avg[ir-rr:].sum()
            rr += 1

    for il in inv_l:
        neighborhood[il] = grad_accu[il,0] / (accus[il,1]*-1 + il + 1)
    for ir in inv_r:
        neighborhood[ir] = grad_accu[ir,1] / (accus[ir,3]*-1 + den_len - ir)

    m_id = np.array(neighborhood).argmax() if mode == 'mo' else np.array(neighborhood).argmin()

    if   accus[m_id, 1] < 0:
        return m_id+1, (1, m_id - int(accus[m_id,1]) + 1), 0
    elif accus[m_id, 3] < 0:
        return m_id+1, (m_id + int(accus[m_id,3]) + 1, den_len), 1
    else:
        return m_id+1, (m_id - int(accus[m_id,1]) + 1, m_id + int(accus[m_id,3]) + 1), 2


def freq_zero_test(model, modelname, benign, benign_fq, grad_fq, f,
                   k, bn, labels, sub=1, method='gradient', mode=modes[0]):
    """Zero-masking in frequency domain: set important band to 0, then IFFT."""
    mixture_fq = benign_fq.copy()
    (tr, ch, ts) = benign.shape

    f_local  = np.fft.fftfreq(ts, d=1/sfreq)
    uq_freqs = np.where(np.isclose(f_local, -1*(sfreq/2), 0.5))[0][0]

    for t in range(tr):
        cid, nb, _ = find_neighbors(benign_fq[t, :, 1:uq_freqs],
                                    abs(grad_fq[t, :, 1:uq_freqs]),
                                    k * 0.01, args.abs_saliency, mode[0])
        # zero out positive-frequency band
        mixture_fq[t, :, nb[0]:nb[1]+1] = 0
        # zero out symmetric negative-frequency band
        mixture_fq[t, :, ts - nb[1] : ts - nb[0] + 1] = 0

    mixture = np.fft.ifft(mixture_fq).real

    _, test_loader_r = get_loader(bn, mixture, labels)
    loss_fn = nn.CrossEntropyLoss()
    if use_auc:
        losses, accs, outs = evaluate_an_epoch_auc(model, device, test_loader_r, loss_fn)
    else:
        losses, accs, outs = evaluate_an_epoch(model, device, test_loader_r, loss_fn)
    return losses, accs, outs


if __name__ == '__main__':
    if use_ssvep_model:
        model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet': SCCNet}
    else:
        model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet': SCCNet}

    leg    = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'inte_grad', 'random']
    filler = '' if args.abs_saliency == 1 else 'n'

    for rep in range(1):
        EXPL_DIR = f"{IRISHSIEH_DIR}/repeat{rep}/expl/{dataname}"
        os.makedirs(os.path.join(SAVE_DIR, f'repeat{rep}/fq_test_zero/{dataname}_si'), exist_ok=True)

        for m in model_type.keys():
            hists, hists1 = [], []

            for l in leg:
                if args.abs_saliency == 1 and l in ['smoothgrad_sq', 'vargrad']:
                    continue
                testmodel = model_type[m](**kwerg)
                print('rep', rep, m, l)

                hist  = dict(acc =np.zeros((n_subs, fq_dim)),
                             loss=np.zeros((n_subs, fq_dim)),
                             out =np.zeros((n_subs, fq_dim, n_trials, n_classes_out)))
                hist1 = dict(acc =np.zeros((n_subs, fq_dim)),
                             loss=np.zeros((n_subs, fq_dim)),
                             out =np.zeros((n_subs, fq_dim, n_trials, n_classes_out)))

                for s in tqdm(range(n_subs)):
                    sub = sub_list[s]

                    model_path = os.path.join(IRISHSIEH_DIR,
                        f"models/{dataname}/bests_repeat{rep}/sub{sub}-{m}.pth")
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    testmodel.load_state_dict(checkpoint["state_dict"])
                    testmodel = testmodel.to(device)

                    mat        = loadmat(os.path.join(DATASET_DIR, mat_file(sub)))
                    xtest_full = mat["x_test"]
                    ytest_full = mat["y_test"].squeeze()
                    xtest      = xtest_full[xtest_sl]
                    ytest      = ytest_full[xtest_sl]

                    if l != 'gradientxinput':
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'),
                                        allow_pickle=True)
                    else:
                        grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'),
                                        allow_pickle=True)
                        grads = np.multiply(grads, xtest_full[gradxi_sl])

                    if args.abs_saliency == 1:
                        grads = np.absolute(grads, out=grads)

                    freqs     = np.fft.fftfreq(xtest.shape[-1], d=1/sfreq)
                    benign_fq = np.fft.fft(xtest)
                    grad_fq   = np.fft.fft(grads)

                    for k in range(1, fq_dim + 1):
                        loss, acc, out = freq_zero_test(
                            testmodel, m, xtest, benign_fq, grad_fq, freqs,
                            k*5, batch_size, ytest, sub, l, mode=modes[0])
                        hist['acc'][s][k-1]  = acc
                        hist['loss'][s][k-1] = loss
                        hist['out'][s][k-1]  = out

                        loss, acc, out = freq_zero_test(
                            testmodel, m, xtest, benign_fq, grad_fq, freqs,
                            k*5, batch_size, ytest, sub, l, mode=modes[1])
                        hist1['acc'][s][k-1]  = acc
                        hist1['loss'][s][k-1] = loss
                        hist1['out'][s][k-1]  = out

                hists.append(hist)
                hists1.append(hist1)

            with open(os.path.join(SAVE_DIR,
                    f'repeat{rep}/fq_test_zero/{dataname}_si/{m}_fq_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as f:
                pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(SAVE_DIR,
                    f'repeat{rep}/fq_test_zero/{dataname}_si/{m}_fq_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as f:
                pickle.dump(hists1, f, protocol=pickle.HIGHEST_PROTOCOL)