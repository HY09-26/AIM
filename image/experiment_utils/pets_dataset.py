import os
from PIL import Image
from torch.utils.data import Dataset

class OxfordPetsDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.transform = transform

        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root, "annotations", split_file)

        self.samples = []
        with open(split_path, "r") as f:
            for line in f:
                name, label, *_ = line.strip().split()
                img_path = os.path.join(root, "images", name + ".jpg")
                self.samples.append((img_path, int(label) - 1))  # label: 0–36

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
