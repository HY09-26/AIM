"""MoRF–LeRF Spearman ρ for image domain.

Thin wrapper around analysis/spearman.py — all logic lives there.

Original usage (unchanged):
    python image/experiment/spearman.py
    python image/experiment/spearman.py --dataset brain_mri --mask pgd
    python image/experiment/spearman.py --out_csv my_spearman.csv
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.spearman import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "image")
    main()
