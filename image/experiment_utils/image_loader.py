import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, file_list, mode="resnet50"):
        """
        file_list: list of (image_path, label)
        mode: 'resnet50' | 'raw'
        """
        self.files = file_list
        self.mode = mode

        # === 對齊 model 的 transform ===
        if mode == "resnet50":
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        elif mode == "raw":
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # img: (3,224,224)
        return img, torch.tensor(label, dtype=torch.long)


def get_loader(
    txt_path,
    batch_size=64,
    shuffle=False,
    mode="resnet50",
    num_workers=4,
):
    file_list = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            # Oxford-IIIT Pet format:
            # image_name class_id species trimap
            img_name = parts[0]
            label = int(parts[1]) - 1   # 1–37 → 0–36

            img_path = os.path.join(
                os.path.dirname(os.path.dirname(txt_path)),  # Oxford_Pet/
                "images",
                img_name + ".jpg"
            )

            file_list.append((img_path, label))

    dataset = ImageDataset(file_list, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader
