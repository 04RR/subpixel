from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from model import Model
from vision.data import ImageDataset

# dataset = torchvision.datasets.FashionMNIST("./", download=True)


class Datas(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __getitem__(self, index):
        return (
            torch.tensor(np.array(self.dataset[index][0])).unsqueeze(0).float().cuda(),
            torch.tensor([1 if i == self.dataset[index][1] else 0 for i in range(10)])
            .float()
            .cuda(),
        )

    def __len__(self):

        return len(self.dataset)


class Test:
    def __init__(self, model, dataset_path, loss_fun, mode, device, transforms=None):

        self.model = model
        self.mode = mode
        self.dataset = ImageDataset(
            dataset_path, mode=mode, device=device, transforms=transforms
        )
        self.loss_fun = loss_fun

    def test(self):

        self.model.fit(self.dataset, self.loss_fun, mode=self.mode, optimizer="adam")



#datase = Datas(dataset_path)
#model = Model().cuda()
#tes = Test(model, datase, loss_fun=nn.MSELoss())
#tes.test()

