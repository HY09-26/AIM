import numpy as np
import os
import math
import random
import pickle#5 as pickle
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

from model import EEGNet, InterpretableCNN, SCCNet
from model import EEGNet_SSVEP, InterpretableCNN_SSVEP
from utils import train_an_epoch_auc, evaluate_an_epoch_auc, train_an_epoch, evaluate_an_epoch, getloader, get_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0" ## adjust according to server
 
def settings(seed): # set random seed by given seed
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


np.set_printoptions(precision=8)
torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64) # np float default: float64

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## initializa reading & saving locations
dataname = 'ERN'
DATASET_DIR = "/mnt/left/home/2023/irishsieh/datasets/P300"
SAVE_DIR = "/mnt/left/home/2023/irishsieh/atk"

## for MI
# kwerg = dict(n_classes=4,
#             channels=22,
#             samples=562,
#             sfreq=125.0)

## for ERN
kwerg = dict(n_classes=2,
            channels=56,
            samples=160,
            sfreq=128)

## for SSVEP
# kwerg = dict(n_classes=5,
#             channels=8,
#             samples=125,
#             sfreq=250,)

            
subs = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26] # ERN

if __name__ == "__main__":
    model_type = {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet':SCCNet}
    # model_type = {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet':SCCNet}

    ## ============ custom variable initializations
    repeats = 5
    epochs = 500
    batch_size = 32
    lr = 1e-4 
    vsplit = (100, 40) # 0.8 #(100, 40) # (100, 100)
    ## ==============

    loss_fn = nn.CrossEntropyLoss()

    # m = 'eegnet'
    # m = 'icnn'
    m = 'sccnet'
    
    accs = np.zeros((16, repeats)) # subject, repeats

    for k in range(16):
        sub = subs[k]
        for j in range(repeats):
            seed = int(np.random.randint(500,size=1)[0])
            settings(seed)

            _, tloader, vloader, testload = getloader(sub, batch_size, vsplit=vsplit, dataset=dataname.lower(), datadir=DATASET_DIR)


            model = model_type[m](**kwerg)
            opt_fn = torch.optim.Adam(model.parameters(), lr=lr)
            # opt_fn = torch.optim.SGD(model.parameters(), lr=lr)
            # sche_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_fn, 'min', factor=lr_exp, cooldown=10)
            sche_fn = torch.optim.lr_scheduler.ExponentialLR(opt_fn, gamma=lr_exp)


            t_loss, t_acc = np.zeros(epochs), np.zeros(epochs)
            for ep in range(epochs):
                loss, acc = train_an_epoch_auc(model.to(device),device, tloader, loss_fn, opt_fn)
                # loss, acc = train_an_epoch(model.to(device),device, tloader, loss_fn, opt_fn)
                with torch.no_grad():
                    val_loss, val_acc, val_out = evaluate_an_epoch_auc(model.to(device),device, vloader, loss_fn) ## ERN
                    test_loss, test_acc, test_out = evaluate_an_epoch_auc(model.to(device),device, testload, loss_fn)
                    # val_loss, val_acc, val_out = evaluate_an_epoch(model.to(device),device, vloader, loss_fn) ## MI, SSVEP
                    # test_loss, test_acc, test_out = evaluate_an_epoch(model.to(device),device, testload, loss_fn) 
                    
                    t_loss[ep] = test_loss
                    t_acc[ep] = test_acc

                ## control learning rate schedule
                if ep>0 and ep%10==0: #@ condition for LR adjustment
                    # sche_fn.step(val_loss) ## some other scheduler takes val_loss as parameter
                    sche_fn.step()
                
                if True:
                    checkpoint = dict(state_dict=model.state_dict(), loss=loss, val_loss=val_loss, bn = batch_size, lr = lr)
                    if not os.path.exists(os.path.join(SAVE_DIR, f'models/{dataname}/sub{sub}')):
                        os.makedirs(os.path.join(SAVE_DIR, f'models/{dataname}/sub{sub}'))
                    torch.save(checkpoint, os.path.join(SAVE_DIR, f"models/{dataname}/sub{sub}/sub{sub}-{m}-ep{ep}.pth"))

            print(f'repeat {j}, sub {sub}, ep {t_acc.argmax()}, acc {t_acc.max()}, seed {seed}')
            ## determine which model is the best
            ## usually people take minimum validation loss as the best
            ## my experiment wants as many saliency maps as possible, so here I took maximum test accuracy
            accs[k, j-1] = t_acc.max()
            best_model_path = os.path.join(SAVE_DIR, "models/{}/sub{}/sub{}-{}-ep{}.pth".format(dataname, sub, sub, m, t_acc.argmax()))
            checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
            torch.save(checkpoint, os.path.join(SAVE_DIR, f"models/{dataname}/bests_repeat{j}/sub{sub}-{m}.pth"))
        print('sub: ', sub, np.round(accs[k]*100, 3))