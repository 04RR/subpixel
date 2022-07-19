import torch
import numpy as np
import torch.nn as nn
import os
import time
import random
import torchvision.transforms as ttf
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import matplotlib.pylab as plt
import warnings

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_grids(image, grid_size, overlap):

    try:
        h, w, _ = image.shape
    except:
        h, w = image.shape

    try:
        h_grid, w_grid = grid_size
    except:
        h_grid, w_grid = grid_size, grid_size

    h_steps = (h - h_grid) // (h_grid * (1 - overlap)) + 1
    w_steps = w / w_grid

    grids = []

    for i in range(int(h_steps)):
        for j in range(int(w_steps)):

            if j == 0:
                w_start = 0
            else:
                w_start = w_grid * (j - overlap)
                w_start = int(np.round(w_start))

            grid = image[i * h_grid : (i + 1) * h_grid, w_start : w_start + w_grid, :]
            grids.append(grid)

    return grids


def read_image(filename, resize=False):

    image = cv2.imread(filename)

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if resize:
        image = cv2.resize(image, resize)

    return image


def display_images(images, nrows=4, ncols=3, cmap=None, title=None):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    if title:
        fig.suptitle(title, fontsize=20)
    c = 0
    for i in range(ncols):
        for j in range(nrows):
            ax[j][i].imshow(images[c], cmap=cmap)
            ax[j][i].axis("off")
            c += 1
    plt.show()


class CPC_Dataset(Dataset):
    def __init__(self, path, grid_size, overlap, transform=None):

        self.path = path
        self.transform = transform
        self.images = os.listdir(path)
        self.images.sort()
        self.grid_size, self.overlap = grid_size, overlap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.path, self.images[idx])
        img = read_image(img_path)
        grids = get_grids(img, self.grid_size, self.overlap)

        if self.transform:
            for i, grid in enumerate(grids):
                grids[i] = self.transform(grid)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 2)):
        super(BasicBlock, self).__init__()

        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride, (1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1))

        self.up = nn.Conv2d(in_channels, out_channels, (1, 1), (2, 2))

    def forward(self, x):

        x_ = self.relu(self.bn(self.conv1(x)))
        x_ = self.bn(self.conv2(x_))

        if self.stride == (2, 2):
            x = self.bn(self.up(x))

        return x_ + x


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.layer1 = nn.ModuleList(
            [BasicBlock(32, 32, stride=(1, 1)), BasicBlock(32, 32, stride=(1, 1))]
        )
        self.layer2 = nn.ModuleList(
            [BasicBlock(32, 64), BasicBlock(64, 64, stride=(1, 1))]
        )
        self.layer3 = nn.ModuleList(
            [BasicBlock(64, 128), BasicBlock(128, 128, stride=(1, 1))]
        )
        self.layer4 = nn.ModuleList(
            [BasicBlock(128, 256), BasicBlock(256, 256, stride=(1, 1))]
        )

    def forward(self, x):

        x = self.maxpool(self.relu(self.bn(self.conv1(x))))

        for layer in self.layer1:
            x = layer(x)

        for layer in self.layer2:
            x = layer(x)

        for layer in self.layer3:
            x = layer(x)

        for layer in self.layer4:
            x = layer(x)

        x = self.avgpool(x)
        return x


# class CPC_Model(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.model = Resnet18()

#         try:
#             self.model = torch.load(r"../input/mri-scan/Encoder_2").to(DEVICE)
#         except:
#             pass

#         self.net = nn.Sequential(
#             nn.Conv2d(256, 128, 1, 1),
#             nn.Conv2d(128, 128, 1, 1),
#             nn.Conv2d(128, 256, 1, 1),
#         )

#     def forward(self, crops):

#         embedding = self.model(crops[0].to(DEVICE))
#         for crop in crops[1:]:
#             emb = self.model(crop.to(DEVICE))
#             embedding = torch.cat([embedding, emb], dim=0)

#         context = embedding.reshape((1, 256, 6, 6))

#         if np.random.rand(1)[0] > 0.5:
#             if np.random.rand(1)[0] > 0.5:
#                 top_half = context[:, :, :3, :]
#                 bottom_half = context[:, :, 3:, :]

#                 return self.net(top_half)

#             else:
#                 bottom_half = context[:, :, 3:, :]
#                 top_half = context[:, :, :3, :]

#                 return self.net(bottom_half)
#         else:
#             if np.random.rand(1)[0] > 0.5:
#                 right_half = context[:, :, :, 3:]
#                 left_half = context[:, :, :, :3]

#                 return self.net(right_half)

#             else:
#                 left_half = context[:, :, :, :3]
#                 right_half = context[:, :, :, 3:]

#                 return self.net(left_half)
