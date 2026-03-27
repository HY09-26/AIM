# experiment_utils/preprocess_chestxray_test.py
# ============================================================
# Preprocess NIH Chest X-ray TEST set
# 8 disease classes + 1 background (No Finding)
# Fixed 500 samples, reproducible
# ============================================================

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
SEED = 43
TEST_SIZE = 500
IMAGE_SIZE = 224

DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]
BACKGROUND_CLASS = "No Finding"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "chestxray")

CSV_PATH = os.path.join(
    DATA_ROOT, "nih-cxr-lt_single-label_balanced_test.csv"
)
IMG_DIR = os.path.join(DATA_ROOT, "images", "images")

OUT_DIR = os.path.join(DATA_ROOT, "preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# Image transform (same as train)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])


def load_image(img_name):
    img_path = os.path.join(IMG_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img.numpy()  # (3,224,224)


# ------------------------------------------------------------
def compute_label(row):
    """
    Return label index:
    0~7 : disease
    8   : No Finding (background)
    """
    vals = row[DISEASE_CLASSES].values
    if vals.sum() >= 1:
        return int(vals.argmax())
    else:
        return len(DISEASE_CLASSES)  # background


def main():
    print(">>> Preprocessing ChestXray TEST ")

    df = pd.read_csv(CSV_PATH)

    # --- compute labels for all test samples ---
    print(">>> Computing labels")
    labels_all = np.array(
        [compute_label(row) for _, row in df.iterrows()],
        dtype=np.int64
    )

    # --- fixed random subset ---
    all_indices = np.arange(len(df))
    chosen = np.random.choice(
        all_indices, size=TEST_SIZE, replace=False
    )

    images = []
    labels = []

    print(">>> Loading images")
    for idx in tqdm(chosen):
        img_name = df.iloc[idx]["id"]
        images.append(load_image(img_name))
        labels.append(labels_all[idx])

    X = np.stack(images)
    y = np.array(labels)

    # --- save ---
    np.save(
        os.path.join(OUT_DIR, "test_images_500.npy"), X
    )
    np.save(
        os.path.join(OUT_DIR, "test_labels_500.npy"), y
    )
    np.save(
        os.path.join(OUT_DIR, "test_indices_500.npy"), chosen
    )

    with open(
        os.path.join(OUT_DIR, "class_names.txt"), "w"
    ) as f:
        for c in DISEASE_CLASSES:
            f.write(c + "\n")
        f.write(BACKGROUND_CLASS + "\n")

    print(">>> Done.")
    print("Saved test_images_500.npy")
    print("Saved test_labels_500.npy")
    print("Saved test_indices_500.npy")


if __name__ == "__main__":
    main()
