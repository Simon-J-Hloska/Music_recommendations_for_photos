import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageEmotionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        target = torch.tensor([row["valence"], row["energy"], row["dance"]], dtype=torch.float32)
        return img, target