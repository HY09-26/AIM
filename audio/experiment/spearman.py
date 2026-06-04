"""MoRF–LeRF Spearman ρ for audio domain.

Thin wrapper around analysis/spearman.py — all logic lives there.

Original usage (unchanged):
    python audio/experiment/spearman.py
    python audio/experiment/spearman.py --dataset esc50 --model cnn14
    python audio/experiment/spearman.py --out_summary summary.csv
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.spearman import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "audio")
    main()
