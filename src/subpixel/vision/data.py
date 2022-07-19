# file that scans for data from ordered folders and generates DataLoader class.
# -----------------------------------------------------------------------------------------------
# Classification - data/train/images, data/train/train_data.csv, data/val/images and data/val/val_data.csv.
# Segmentation - data/train/images, data/train/masks, data/train/train_data.csv, data/val/images, data/val/masks and data/val/val_data.csv.
# Object Detection - data/train/images, data/train/train_data.csv, data/val/images and data/val/val_data.csv.
# bboxes - [x, y, h, w, classes]

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from PIL import Image
from utils import *
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


class ImageDataset(Dataset):
    '''
    Class that takes in the path of the dataset and converts it into a torch.utils.data.Dataset object.
    '''
    def __init__(self, path, mode, device, transforms=None, train=True):
        super().__init__()

        self.transforms, self.mode, self.device = transforms, mode, device
        self.path = path

        self.df = pd.read_csv(f"{self.path}\\data.csv")[:1]

        if mode == "classification":
            self.classes = self.df["class"].unique()
            self.df[self.classes] = pd.get_dummies(self.df["class"])
            del self.df["class"]

        if mode == "detection":
            for i in range(len(self.df)):
                self.df["labels"].iloc[i] = get_boxxes(self.df["labels"].iloc[i])

    def __getitem__(self, idx):

        img_path = f"{self.path}\\images\\" + self.df["img_path"].iloc[idx]
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.mode == "classification":

            label = torch.tensor(np.array(self.df[self.classes].iloc[idx]))

            if self.transforms:

                transformed = self.transforms(image=img)
                img = transformed["image"]

        elif self.mode == "detection":

            label = np.array(self.df["labels"].iloc[idx])

            if self.transforms:

                transformed = self.transforms(image=img, bboxes=label)
                img = transformed["image"]
                label = transformed["bboxes"]

            label = torch.tensor(label)

        elif self.mode == "segmentation":

            img_path = f"{self.path}\\masks\\" + self.df["mask_path"].iloc[idx]
            label = np.array(Image.open(img_path).convert("RGB"))

            if self.transforms:

                transformed = self.transforms(image=img, mask=label)
                img = transformed["image"]
                label = transformed["mask"]

            label = torch.tensor(label).permute(2, 0, 1)

        return (
            torch.tensor(img).permute(2, 0, 1).float().to(self.device),
            label.float().to(self.device),
        )

    def __len__(self):
        return len(self.df)


def get_dataset(path, mode, device, transforms=None):
    '''
    Function that takes in the path and generates a trainset and valset (if present).

    path: str 
    mode: str
    device: str
    transforms: albumentations.transforms

    Returns trainset and valset
    '''

    trainset = ImageDataset(
        f"{path}train\\", mode, device, transforms=transforms, train=True
    )
    try:
        valset = ImageDataset(f"{path}\\val\\", mode, device, train=False)
        return trainset, valset
    except FileNotFoundError:
        return trainset

def get_dataloader(datset, b_size, shuffle):
    '''
    Converts the dataset to a DataLoader.

    dataset: torch.utils.data.Dataset

    Returns torch.utils.data.DataLoader
    '''

    return DataLoader(datset, b_size, shuffle)
