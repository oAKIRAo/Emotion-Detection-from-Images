import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AffectNetDataset(Dataset):
    emotion2idx = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "surprise": 3,
        "fear": 4,
        "disgust": 5,
        "anger": 6
    }

    def __init__(self, csv_file=None, img_dir=None, transform=None, df=None):
        if df is not None:
            self.data = df.reset_index(drop=True)
        else:
            self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, *row['pth'].split('/'))
        if not os.path.exists(img_path):
            print("⚠️ Image not found:", img_path)
        image = Image.open(img_path).convert("RGB")
        label = self.emotion2idx[row['label']]
        if self.transform:
            image = self.transform(image)
        return image, label
