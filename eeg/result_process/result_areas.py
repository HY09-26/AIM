"""Area metrics for EEG domain.

Thin wrapper around analysis/area.py — all logic lives there.

Original usage (unchanged):
    python eeg/result_process/result_areas.py
    python eeg/result_process/result_areas.py --eeg_domain ch
    python eeg/result_process/result_areas.py --dataset MI --model eegnet
    python eeg/result_process/result_areas.py --out_csv my_eeg_results.csv
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.area import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "eeg")
    main()
