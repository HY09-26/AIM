"""
Per-dataset configuration for EEG faithfulness experiments.

Usage:
    from experiment_utils.eeg_config import get_config, IRISHSIEH_DIR, SAVE_DIR, mat_file
    cfg = get_config('ERN')
    # cfg.kwerg, cfg.sub_list, cfg.dataset_dir, cfg.use_auc, ...

Note: SSVEP data only has repeats 1 and 2; pass --rep 1 or --rep 2 for SSVEP.
"""
from types import SimpleNamespace

# ===== ADJUST THESE TWO PATHS FOR YOUR SETUP =====
IRISHSIEH_DIR = "/mnt/right/alumni/2023/irishsieh/atk"
SAVE_DIR      = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"
# ==================================================

_CONFIGS = {
    'MI': dict(
        dataset_dir     = "/mnt/right/alumni/2023/irishsieh/datasets/MI",
        kwerg           = dict(n_classes=4, channels=22, samples=562, sfreq=125.0),
        use_ssvep_model = False,
        sub_list        = list(range(1, 10)),
        n_trials        = 288,
        n_classes_out   = 4,
        xtest_sl        = slice(None),
        gradxi_sl       = slice(None),
        use_auc         = False,
        batch_size      = 32,
        sfreq           = 125,
        lowfq           = 19,
        highfq          = 176,
    ),
    'ERN': dict(
        dataset_dir     = "/mnt/right/alumni/2023/irishsieh/datasets/P300",
        kwerg           = dict(n_classes=2, channels=56, samples=160, sfreq=128),
        use_ssvep_model = False,
        sub_list        = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
        n_trials        = 40,
        n_classes_out   = 2,
        xtest_sl        = slice(-40, None),
        gradxi_sl       = slice(-40, None),
        use_auc         = True,
        batch_size      = 25,
        sfreq           = 128,
        lowfq           = 1,
        highfq          = 51,
    ),
    'SSVEP': dict(
        dataset_dir     = "/mnt/right/alumni/2023/irishsieh/datasets/MAMEM",
        kwerg           = dict(n_classes=5, channels=8, samples=125, sfreq=250),
        use_ssvep_model = True,
        sub_list        = list(range(1, 12)),
        n_trials        = 100,
        n_classes_out   = 5,
        xtest_sl        = slice(-100, None),
        gradxi_sl       = slice(-100, None),
        use_auc         = False,
        batch_size      = 25,
        sfreq           = 125,   # FFT uses 125 Hz (not the 250 Hz recording rate)
        lowfq           = 1,
        highfq          = 41,
    ),
}


def mat_file(dataname: str, sub: int) -> str:
    """Return the filename of the raw .mat file for a given dataset and subject."""
    if dataname == 'MI':
        return f"BCIC_S{sub:02d}_E.mat"
    elif dataname == 'ERN':
        return f"Data_S{sub:02d}_Sess.mat"
    elif dataname == 'SSVEP':
        return f"U0{sub:02d}.mat"
    raise ValueError(f"Unknown dataset: {dataname!r}. Choose from {list(_CONFIGS)}")


def get_config(dataname: str) -> SimpleNamespace:
    """Return a SimpleNamespace with all per-dataset parameters."""
    if dataname not in _CONFIGS:
        raise ValueError(f"Unknown dataset: {dataname!r}. Choose from {list(_CONFIGS)}")
    return SimpleNamespace(**_CONFIGS[dataname])
