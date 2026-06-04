"""Area metrics for audio domain.

Thin wrapper around analysis/area.py — all logic lives there.

Original usage (unchanged):
    python audio/experiment/area.py
    python audio/experiment/area.py --dataset audiomnist --model audionet
    python audio/experiment/area.py --dataset esc50 --mask pgd
    python audio/experiment/area.py --out_csv my_results.csv
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.area import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "audio")
    main()
