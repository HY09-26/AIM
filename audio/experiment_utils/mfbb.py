#mfbb.py

import numpy as np
import os
import math
import random

## a lot of variables are written fixed according to my experiment for faster computation
## for extended usage, either check github or fix the parts commented out in atk/ directory (better both)


seed = int(0) 
np.random.seed(seed)
random.seed(seed)

def autocov(t, H2):
    return 0.5*(math.fabs(t-1)**H2 + (t+1)**H2 - 2*t**H2)

def cov(t0, t1, H2, scalar):
    res = scalar*(t0**H2 + t1**H2 - math.fabs(t0-t1)**H2)
    if res < 1e-14:
        res += 1e-10
    return res


def fit_hurst(signal):
    signal = norm(signal)
    SF2 = ((signal-signal[0])**2).sum()/(signal.shape[0]-1)
    tao = 1/signal.shape[0]
    return math.log(SF2)/(2*math.log(tao))


def norm(arr):
   return (arr-arr.min())/(arr.max()-arr.min()+(1e-15))

def FBM(H, N, method="DaviesHarte", stddev = 1):
    # y0, y1: start/end
    # H: hurst
    # N: point count for generated
    # T: 

    if method == "DaviesHarte":
        M = np.sqrt(2 * N)
        # Circulant autocov
        c = [autocov(t, H*2) for t in range(N)]
        c = c + [0] + np.flip(c[1:]).tolist()
        
        # eiganvalues
        Lambda = np.sqrt(np.fft.fft(c).real)
        # print(Lambda.shape, N)

        # Z, Z2 = np.random.standard_normal(N), np.random.standard_normal(N)
        Z, Z2 = np.random.normal(0, stddev, N), np.random.normal(0, stddev, N)

        # Q*Z
        qZ = np.zeros(2 * N, dtype=complex)

        qZ[0]   = Lambda[0]/M * Z[0]
        qZ[1:N] = Lambda[1:N]/(np.sqrt(2)*M) * (Z[1:] + 1j * Z2[1:])
        qZ[N]   = Lambda[0]/M * Z2[0]
        qZ[N+1:]  = Lambda[N+1:]/(np.sqrt(2)*M) * (np.flip(Z[1:]) + 1j * np.flip(Z2[1:]))

        # QL^1/2Q*Z = SZ, fgn
        SZ = np.fft.fft(qZ).real[:N]

        fbm = np.cumsum(SZ)*(N**-H)
        return fbm

    elif method == "Hosking":
        raise NotImplementedError
    elif method == "Cholesky":
        raise NotImplementedError
    else:
        raise NotImplementedError


def MFBB(XI, Xt, T, H, N, original_std=None, method="DaviesHarte"):
    # XI, TI: points conditioned on
    # TI: times [0, 0.5, 1]
    # Xt, T: final


    TI = np.array([0, 0.5, 1])
    Xbt = np.zeros(2*N)

    fbm_whole = FBM(H, N*2-1, method, original_std) 
    fbm_whole -= fbm_whole.mean()
    fbm_whole = np.insert(fbm_whole, [0], XI[0])
    

    gamma_param = 1-2*H
    if gamma_param < 1e-14:
        gamma_param +=  1e-10

    scalar = (math.gamma(gamma_param)*math.cos(H*math.pi))/2*H*math.pi

    for i in range(TI.shape[0]-1):
        fbm = fbm_whole[i*N:(i+1)*N]
        
        ti = TI[i]
        TJ = np.linspace(ti, TI[i+1], N)

        covij, covTj = np.zeros(N), np.zeros(N)
        for j in range(N):
            tj = TJ[j]
            covij[j] = cov(ti, tj, H*2, scalar)
            covTj[j] = cov(T, tj, H*2, scalar)
        Xbt[i*N:(i+1)*N] = fbm - ((fbm_whole[int(ti)]-XI[i])/covij) * covTj
    return Xbt, fbm_whole

