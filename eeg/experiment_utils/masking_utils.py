"""
Shared masking utilities for EEG faithfulness experiments.

  find_neighbors       – Contiguous frequency-band selection (fq*_test.py)
  find_crop            – Sliding time-window selection (ts*_test.py)
  NoisySpatialImputer  – Spatial interpolation imputer (chroad_test.py)
  MI / ERN / SSVEP     – Electrode topology definitions
  get_topology         – Convenience factory for topology objects
"""

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


# ── Frequency masking ─────────────────────────────────────────────────────────

def find_neighbors(den: np.ndarray, grad: np.ndarray, ratio: float, mode: str):
    """
    Find the contiguous frequency band that accounts for `ratio` of total
    spectral energy and contains the most-salient (mode='mo') or
    least-salient (mode='le') frequency.

    Args:
        den:   Spectral density, shape (n_ch, n_bins) or (n_bins,).
        grad:  Saliency in frequency domain, same shape as den.
        ratio: Target energy fraction in (0, 1].
        mode:  'mo' → most-salient band; 'le' → least-salient band.

    Returns:
        (center_idx, (left_idx, right_idx), boundary_case)
    """
    den_avg  = np.abs(den).mean(axis=0)  if den.ndim  > 1 else np.abs(den)
    grad_avg = np.abs(grad).mean(axis=0) if grad.ndim > 1 else np.abs(grad)
    n = den_avg.shape[-1]

    target    = den_avg.sum() * ratio
    target_hv = target / 2

    # accus columns: left_sum, left_reach, right_sum, right_reach
    accus = np.ones((n, 4)) * -n
    accus[:, 0], accus[:, 2] = den_avg, den_avg
    accus[den_avg >= target, 1] = 0
    accus[den_avg >= target, 3] = 0

    grad_accu = np.zeros((n, 2))
    grad_accu[:, 0] = grad_avg

    for i in range(n - 1):
        ls = np.logical_and(accus[i+1:, 0] < target_hv, accus[i+1:, 1] < 0)
        accus[i+1:, 0][ls]     += den_avg[:n-i-1][ls]
        grad_accu[i+1:, 0][ls] += grad_avg[:n-i-1][ls]
        accus[np.logical_and(accus[:, 0] >= target_hv, accus[:, 1] < 0), 1] = i + 1

        rs = np.logical_and(accus[:n-i-1, 2] < target_hv, accus[:n-i-1, 3] < 0)
        accus[:n-i-1, 2][rs]     += den_avg[i+1:][rs]
        grad_accu[:n-i-1, 1][rs] += grad_avg[i+1:][rs]
        accus[np.logical_and(accus[:, 2] >= target_hv, accus[:, 3] < 0), 3] = i + 1

    valid = np.logical_and(accus[:, 1] >= 0, accus[:, 3] >= 0)
    neighborhood = np.zeros(n)
    neighborhood[valid] = (grad_accu.sum(axis=1)[valid]
                           / (accus[valid, 1] + accus[valid, 3] + 1))

    inv_l = np.where(accus[:, 1] < 0)[0]
    inv_r = np.where(accus[:, 3] < 0)[0]

    for il in inv_l:
        ll = 1
        while accus[il, 0] < target and il + ll < n:
            accus[il, 0]     = den_avg[:il+ll].sum()
            accus[il, 1]     = -ll
            grad_accu[il, 0] = grad_avg[:il+ll].sum()
            ll += 1
    for ir in inv_r:
        rr = 1
        while accus[ir, 2] < target and rr <= ir:
            accus[ir, 2]     = den_avg[ir-rr:].sum()
            accus[ir, 3]     = -rr
            grad_accu[ir, 1] = grad_avg[ir-rr:].sum()
            rr += 1

    for il in inv_l:
        neighborhood[il] = grad_accu[il, 0] / (-accus[il, 1] + il + 1)
    for ir in inv_r:
        neighborhood[ir] = grad_accu[ir, 1] / (-accus[ir, 3] + n - ir)

    m_id = neighborhood.argmax() if mode == 'mo' else neighborhood.argmin()

    if accus[m_id, 1] < 0:
        return m_id + 1, (1, m_id - int(accus[m_id, 1]) + 1), 0
    elif accus[m_id, 3] < 0:
        return m_id + 1, (m_id + int(accus[m_id, 3]) + 1, n), 1
    else:
        return m_id + 1, (m_id - int(accus[m_id, 1]) + 1, m_id + int(accus[m_id, 3]) + 1), 2


# ── Time-segment masking ──────────────────────────────────────────────────────

def find_crop(grad: np.ndarray, ratio: float = 0.1):
    """
    Find start indices and window size for the most- and least-salient
    contiguous time window of width `ratio * T`.

    Args:
        grad:  Saliency, shape (..., n_ch, n_ts). Leading dims are averaged.
        ratio: Window width as a fraction of the time axis length.

    Returns:
        (max_start, min_start, window_size)
    """
    grad = grad.squeeze()
    if grad.ndim > 2:
        grad = grad.mean(axis=0)   # → (n_ch, n_ts)

    wsize  = int(ratio * grad.shape[-1])
    winsum = np.convolve(grad[0], np.ones(wsize), 'valid')
    for c in range(1, grad.shape[0]):
        winsum += np.convolve(grad[c], np.ones(wsize), 'valid')

    return int(winsum.argmax()), int(winsum.argmin()), wsize


# ── Spatial (channel) masking ─────────────────────────────────────────────────

# Direct-neighbour / indirect-neighbour weighting coefficients
_CH_WEIGHTS = (1 / 6, 1 / 12)


class _EEGTopology:
    """Base container for EEG electrode neighbourhood topology."""
    ch_name: list
    dn_id:   list   # direct-neighbour offsets per channel
    idn_id:  list   # indirect-neighbour offsets per channel


class MI(_EEGTopology):
    """BCI Competition IV 2a motor-imagery — 22 channels."""
    def __init__(self):
        self.ch_name = [
            'Fz',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz',
        ]
        self.dn_id = [
            (3,),
            (1, 6), (-1, 1, 6), (-1, 1, -3, 6), (-1, 1, 6), (1, 6),
            (1,), (-1, 1, -6, 7), (-1, 1, -6, 7), (-1, 1, -6, 7),
            (-1, 1, -6, 7), (-1, 1, -6, 7), (-1,),
            (-6, 1), (-1, 1, -6, 4), (-1, 1, -6, 4), (-1, 1, -6, 4), (-1, -6),
            (-4, 1), (-1, 1, -4, 2), (-14, -1), (-2,),
        ]
        self.idn_id = [
            (2, 4),
            (5, 7), (5, 7), (5, 7), (5, 7), (5, 7),
            (-5, 7), (-5, 7), (-7, -5, 5, 7), (-7, -5, 5, 7),
            (-7, -5, 5, 7), (-7, 5), (-7, 5),
            (-7, -5, 5), (-7, -5, 5), (-7, -5, 3, 5), (-7, -5, 3), (-7, -5, 3),
            (-5, -3, 3), (-5, -3), (-5, -3, 1), (-3, -1),
        ]


class ERN(_EEGTopology):
    """BCI Challenge ERN — 56 channels."""
    def __init__(self):
        self.ch_name = [
            'Fp1', 'Fp2',
            'AF7', 'AF3', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'POz', 'PO8',
            'O1', 'O2',
        ]
        self.dn_id = [
            (3, 4), (3, 4),
            (1, 5), (-3, -1, 5), (-3, 1, 8), (-1, 8),
            (1, 9), (-5, -1, 1, 9), (-5, -1, 1, 9), (-1, 1, 9), (-1, 1, 9),
            (-1, 1, 9), (-8, -1, 1, 9), (-8, -1, 1, 9), (-1, 9),
            (-9, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9),
            (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 9),
            (-9, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9),
            (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 9),
            (-9, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9),
            (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 1, 9), (-9, -1, 9),
            (-9, 1), (-9, -1, 1, 8), (-9, -1, 1), (-9, -1, 1), (-9, -1, 1, 6),
            (-9, -1, 1), (-9, -1, 1), (-9, -1, 1, 4), (-9, -1),
            (-8, 3), (-6,), (-4, 2),
            (-3, 1), (-2, -1),
        ]
        self.idn_id = [
            (1, 2), (-1, 2),
            (-2, 4, 6), (4, 6), (7, 9), (-4, 7, 9),
            (-4, 10), (-4, 8, 10), (-6, 8, 10), (-6, 8, 10), (8, 10),
            (-7, 8, 10), (-7, 8, 10), (-9, 8, 10), (-9, 8),
            (-8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10),
            (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, 8),
            (-8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10),
            (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, 8),
            (-8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10),
            (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, -8, 8, 10), (-10, 8),
            (-8, 9), (-10, -8), (-10, -8, 7), (-10, -8, 7), (-10, -8),
            (-10, -8, 5), (-10, -8, 5), (-10, -8), (-10, 4),
            (-9, -7), (-7, -5), (-5, -3),
            (-2,), (-3,),
        ]


class SSVEP(_EEGTopology):
    """MAMEM SSVEP — 8 channels."""
    def __init__(self):
        self.ch_name = ['PO7', 'PO3', 'O1', 'POz', 'Oz', 'PO4', 'O2', 'PO8']
        self.dn_id  = [(1,), (-1, 1, 2), (-1, 2), (-2, 2, 1), (-2, 2, -1), (-2, 1, 2), (-2, -1), (-2,)]
        self.idn_id = [(2,), (3,), (-2, 1), (-1, 3), (-3, 1), (-1,), (-3, 1), (-1,)]


_TOPOLOGIES = {'MI': MI, 'ERN': ERN, 'SSVEP': SSVEP}


def get_topology(dataname: str) -> _EEGTopology:
    """Return the electrode-topology object for the given dataset."""
    if dataname not in _TOPOLOGIES:
        raise ValueError(f"No topology defined for {dataname!r}")
    return _TOPOLOGIES[dataname]()


class NoisySpatialImputer:
    """
    Replace a set of EEG channels by spatially-interpolated values plus
    small Gaussian noise, using the electrode neighbourhood topology.

    Args:
        mask:     Channel indices to impute.
        topology: An _EEGTopology instance (MI / ERN / SSVEP).
        noise:    Std-dev of additive Gaussian noise after interpolation.
    """

    def __init__(self, mask: list, topology: _EEGTopology, noise: float = 0.01):
        self.topology   = topology
        self.noise      = noise
        self.imputed_id = mask
        self.n          = len(mask)

        # valid[i] = 0 if channel i has any neighbour that is also imputed
        self.valid = np.ones(self.n)
        for i, ch in enumerate(self.imputed_id):
            nb = np.array(list(topology.dn_id[ch]) + list(topology.idn_id[ch])) + ch
            if np.any(np.isin(nb, self.imputed_id)):
                self.valid[i] = 0

    def _neighbours(self, ch: int) -> list:
        dn  = self.topology.dn_id[ch]
        idn = self.topology.idn_id[ch]
        dnw  = 4 / len(dn)  * _CH_WEIGHTS[0]
        idnw = 4 / len(idn) * _CH_WEIGHTS[1]
        return [(dnw, dn_i + ch) for dn_i in dn] + [(idnw, idn_i + ch) for idn_i in idn]

    def _build_system(self, trial: np.ndarray):
        c2v = np.zeros(trial.shape[0], dtype=np.int32)
        c2v[self.imputed_id] = np.arange(self.n)

        A    = lil_matrix((self.n, self.n))
        b    = np.zeros((self.n, trial.shape[1]))
        diag = np.ones(self.n)

        for i, ch in enumerate(self.imputed_id):
            for w, nb_ch in self._neighbours(ch):
                b[i] -= w * trial[nb_ch]
            if not self.valid[i]:
                A[i, c2v[i]] = w
                diag[i] -= w

        A[np.arange(self.n), np.arange(self.n)] = -diag
        return A, b

    def impute(self, trial: np.ndarray) -> np.ndarray:
        """Return trial with masked channels replaced by interpolated values."""
        result    = trial.copy()
        A, b      = self._build_system(trial)
        x         = np.array(spsolve(csc_matrix(A), b))
        x        += np.random.randn(*x.shape) * self.noise
        result[self.imputed_id] = x
        lo, hi    = trial.min(), trial.max()
        result    = (result - result.min()) / (result.max() - result.min() + 1e-15)
        result    = result * (hi - lo) + lo
        return result
