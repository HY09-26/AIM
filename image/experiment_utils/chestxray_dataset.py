# experiment_utils/chestxray_dataset.py

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayTrainDataset(Dataset):
    """
    NIH Chest X-ray
    8 disease classes + 1 background (No Finding)
    Read from train_label_nf10k.txt (id + label_name)
    """

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

    def __init__(self, label_txt_path, img_dir):
        print(">>> ChestXrayTrainDataset: loading label txt")
        print(f">>> label file = {label_txt_path}")

        self.img_dir = img_dir

        # --------------------------------------------------
        # label name -> index mapping
        # --------------------------------------------------
        self.class_names = self.DISEASE_CLASSES + [self.BACKGROUND_CLASS]
        self.class_to_idx = {
            name: i for i, name in enumerate(self.class_names)
        }

        # --------------------------------------------------
        # load label txt
        # --------------------------------------------------
        self.samples = []  # list of (img_id, label_idx)

        with open(label_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id, label_name = line.split(" ", 1)

                if label_name not in self.class_to_idx:
                    raise ValueError(
                        f"Unknown label name: {label_name}"
                    )

                label_idx = self.class_to_idx[label_name]
                self.samples.append((img_id, label_idx))

        self.num_classes = len(self.class_names)

        print(f">>> Num samples : {len(self.samples)}")
        print(f">>> Num classes : {self.num_classes}")
        print(">>> Class mapping:")
        for k, v in self.class_to_idx.items():
            print(f"    {v}: {k}")

        # --------------------------------------------------
        # transform (keep it simple & fast)
        # --------------------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label
