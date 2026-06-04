"""MoRF/LeRF curve plots for image domain.

Thin wrapper around analysis/plot_morf_lerf.py — all logic lives there.

Original usage (unchanged):
    python image/experiment/plot_morf_lerf.py
    python image/experiment/plot_morf_lerf.py --datasets brain_mri --models resnet_50
    python image/experiment/plot_morf_lerf.py --masks pgd road
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.plot_morf_lerf import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "image")
    main()
