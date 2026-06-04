"""MoRF/LeRF curve plots for audio domain.

Thin wrapper around analysis/plot_morf_lerf.py — all logic lives there.

Original usage (unchanged):
    python audio/experiment/plot_morf_lerf.py
    python audio/experiment/plot_morf_lerf.py --datasets esc50 --esc50_folds 1 2 3
    python audio/experiment/plot_morf_lerf.py --models audionet --masks pgd
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.plot_morf_lerf import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "audio")
    main()
