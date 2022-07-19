import json
import numpy
from torch.utils.data import DataLoader
import torch
import torchvision
from numpy import diff
from tqdm import tqdm
import torch.nn as nn
import os
import numpy as np
import random


def get_boxxes(t):
    # '{x, y, h, w, [classes]}' -> [x, y, h, w, classes]
    bbox = list(json.loads(t).values())
    return bbox[:-1] + bbox[-1]


def seed_everything(seed=42):
    """
    Seeds EVERYTHING.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_model(m):
    """
    Initialises model parameters with xavier normalisation method.
    """

    seed_everything()

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight.data)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)


def findLR(
    model: nn.Module,
    dataset: nn.Module,
    loss_fn: nn.Module,
    optimizer: str,
    start_lr: float = 1e-7,
    end_lr: float = 1e-1,
    steps: float = 100,
):
    """
    Finds the ideal initial LR for optimal training.
    model : nn.Module , the model for which ideal LR needs to be found.

    dataset : nn.Module ,  the dataset to be used to train.

    loss_fn : nn.Module , loss to measure model.

    optimizer : str , lowercase string of preferred optimizer.

    start_lr : lower bound of the learning rate to be checked.

    end_lr : upper bound of the learning rate to be checked.

    steps : number of learning rates between start_lr and end_lr to be checked. 
    """
    seed_everything()
    lr = []
    loss = []
    optimizer = get_optimizer(model, lr=start_lr)
    dx = (end_lr - start_lr) / steps

    x = find_batch_size(model, dataset)
    if len(dataset) // steps < x:
        x = len(dataset) // steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch + dx)
    Dataloader = iter(DataLoader(dataset, x, True))
    model.train()

    model = model.cuda()
    model.apply(init_model)

    for i in tqdm(range(0, steps)):

        data, label = next(Dataloader)
        pred = model(data)
        loss_ = loss_fn(pred, label)

        loss.append(loss_.detach().cpu().numpy())
        lr.append(start_lr + i * dx)
        optimizer.zero_grad()

        loss_.backward()
        optimizer.step()
        scheduler.step()

    model.apply(init_model)

    return lr[numpy.argmin(diff(loss) / dx)], loss, lr


def find_batch_size(model: nn.Module, dataset: nn.Module):
    """
    Finds the batch size to be set for ideal GPU usage (95% default)

    model : nn.Module , model being trained.

    dataset : nn.Module , dataset to be loaded.

    """

    p, total_bits = model.find_size()
    f_before = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)

    for data in dataset:
        img, label = data
        img = img.cuda()
        label = label.cuda()
        break

    f_after = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    data_size = -f_after + f_before

    available_size = 0.95 * (f_after - total_bits + data_size)

    torch.cuda.empty_cache()
    b_size = int(available_size // data_size)
    return b_size


def get_optimizer(
    model: nn.Module, optim: str = "adam", lr: float = 1e-3, weight_decay: float = 1e-5
):
    """
    returns torch.optim optimizer instance given optim string
    """
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer not implemented yet!!")
