from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from config import IMG_SIZE


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15,
            hue=0.05
        ),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5)
        ], p=0.2),

        transforms.RandomAutocontrast(p=0.2),

        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3)
        )
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


class MuzzleDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None
    ):

        self.df = pd.read_csv(csv_path)

        self.transform = transform

        unique_ids = sorted(self.df["godhaarId"].unique())

        self.label_map = {
            cid: idx
            for idx, cid in enumerate(unique_ids)
        }

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        img_path = Path(row["local_path"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[row["godhaarId"]]

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "godhaar_id": row["godhaarId"],
            "path": str(img_path)
        }

    @property
    def num_classes(self):
        return len(self.label_map)