"""Area metrics for image domain.

Thin wrapper around analysis/area.py — all logic lives there.

Original usage (unchanged):
    python image/experiment/area.py
    python image/experiment/area.py --dataset imagenet --model resnet_50
    python image/experiment/area.py --out_csv my_results.csv
"""
import sys
import os

# Make analysis/ importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.area import main  # noqa: E402

if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.insert(1, "--domain")
        sys.argv.insert(2, "image")
    main()
