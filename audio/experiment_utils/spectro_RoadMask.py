import numpy as np

import numpy as np
from scipy.signal import convolve2d


def laplacian_interpolate_pixels(
    spec2d,
    alpha=0.1,
    clip_min=None,
    clip_max=None,
    boundary="symm",
):
    """
    Laplacian interpolation with optional Gaussian noise.

    Parameters
    ----------
    spec2d : np.ndarray, shape (H, W)
        Input spectrogram.
    alpha : float
        Noise strength. 0.0 = no noise.
        Recommended: 0.05 ~ 0.15
    clip_min, clip_max : float or None
        Clipping range (e.g., spectrogram dB range).
    boundary : str
        Boundary handling for convolution.

    Returns
    -------
    out : np.ndarray, shape (H, W)
        Interpolated spectrogram.
    """

    assert spec2d.ndim == 2, f"Expected 2D input, got {spec2d.shape}"

    # 3×3 Laplacian kernel
    kernel = np.array([
        [1/12, 1/6, 1/12],
        [1/6,   0.0, 1/6],
        [1/12, 1/6, 1/12],
    ], dtype=np.float32)

    # Laplacian interpolation (C-level, fast)
    interp = convolve2d(
        spec2d,
        kernel,
        mode="same",
        boundary=boundary
    )

    # Optional noise (Ref-ROAD style)
    if alpha > 0.0:
        sigma = np.std(spec2d)
        noise = np.random.normal(
            loc=0.0,
            scale=alpha * sigma,
            size=interp.shape
        )
        interp = interp + noise

    # Optional clipping
    if clip_min is not None or clip_max is not None:
        interp = np.clip(interp, clip_min, clip_max)

    return interp


def compute_patch_saliency(saliency_2d, patch_size):
    """
    saliency_2d: (H, W)
    return:
        patch_scores: (n_patches,)
        patch_indices: list of (i0, i1, j0, j1)
    """
    H, W = saliency_2d.shape
    ps = patch_size

    patch_scores = []
    patch_indices = []

    for i in range(0, H, ps):
        for j in range(0, W, ps):
            i1 = min(i + ps, H)
            j1 = min(j + ps, W)

            patch = saliency_2d[i:i1, j:j1]
            score = np.mean(np.abs(patch))  # patch-level attribution

            patch_scores.append(score)
            patch_indices.append((i, i1, j, j1))

    return np.array(patch_scores), patch_indices
