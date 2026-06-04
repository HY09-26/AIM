"""MoRF–LeRF Spearman ρ for EEG domain.

Thin wrapper around analysis/spearman.py — all logic lives there.

Original usage (unchanged):
    python eeg/result_process/result_spears.py
    python eeg/result_process/result_spears.py --eeg_domain ts
    python eeg/result_process/result_spears.py --dataset ERN --model eegnet
    python eeg/result_process/result_spears.py --out_csv my_eeg_spearman.csv
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.spearman import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "eeg")
    main()
