from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import warnings
from data import ImageDataset, get_dataloader, get_dataset
import numpy as np
import torch.nn as nn
from utils import findLR, find_batch_size, get_optimizer


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(out: torch.Tensor, labels: torch.Tensor):
    """
    Finds the accuracy of the model by comparing the output of the model to the labels.

    out: tensor
    labels: tensor
    """
    try:
        return (out == labels).sum().item() / out.size(0) * out.size(1) * out.size(2)
    except:
        return (out == labels).sum().item() / out.size(0) * out.size(1)


class visionTrainer:
    """
    class that has all the funcions and variables to train a model on your custom dataset.

    model: nn.Module
    trainset: str or (Dataset, ImageDataset)
    transforms: 
    optimizer: str
    valset: (Dataset, ImageDataset)
    epochs: int
    mode: str ["classification", "detection", "segmentation"]
    loss_fn: nn.Module
    learning_rate: float
    weight_decay: float
    model_save_path: str
    shuffle: bool
    device: str ["cpu", "cuda"]
    """

    def __init__(
        self,
        model,
        trainset,
        transforms=None,
        optimizer="adam",
        valset=None,
        epochs=10,
        mode="classification",
        loss_fn=nn.MSELoss(),
        learning_rate=None,
        weight_decay=1e-5,
        model_save_path="./",
        shuffle=True,
        device="cpu",
    ):
        self.model = model.cuda() if device == "cuda" else model
        self.valset = valset
        self.epochs = epochs
        self.mode = mode
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.model_save_path = model_save_path
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.device = device

        if isinstance(trainset, str):
            try:
                self.trainset, self.valset = get_dataset(
                    trainset, self.mode, device, transforms
                )
            except:
                self.trainset = get_dataset(trainset, self.mode, device, transforms)

        elif isinstance(trainset, Dataset) or isinstance(trainset, ImageDataset):
            self.trainset = trainset
            self.valset = valset

        self.b_size = find_batch_size(model, self.trainset)

        if learning_rate == None:
            self.learning_rate = findLR(
                self.model, self.trainset, self.loss_fn, optimizer
            )[0]

        self.optimizer = get_optimizer(
            self.model,
            optim=optimizer,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.train_dl = get_dataloader(self.trainset, self.b_size, self.shuffle)

        if self.valset != None:
            self.val_dl = get_dataloader(self.valset, self.b_size, self.shuffle)

    def fit(self):
        """
        Function that has the training loop implemented. 
        It inherits all the necessary components from the Trainer class.

        Returns the loss values and acc values if applicable. 
        """

        flag = self.mode == "classification" or self.mode == "detection"
        scaler = torch.cuda.amp.GradScaler()
        losses = {"train": [], "val": []}
        acc = {"train": [], "val": []}

        for epoch in range(self.epochs):

            epoch_loss = {"train": [], "val": []}
            epoch_acc = {"train": [], "val": []}

            self.model.train()
            for img, label in tqdm(self.train_dl):

                with torch.cuda.amp.autocast():

                    pred = self.model(img)
                    loss = self.loss_fn(pred, label)

                    epoch_loss["train"].append(loss)

                    if self.mode == "classification":
                        a = accuracy(pred, label)
                        epoch_acc["train"].append(a)

                    elif self.mode == "detection":
                        a = accuracy(pred[1:5], label[1:5])
                        epoch_acc["train"].append(a)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            losses["train"].append(sum(epoch_loss["train"]) / len(epoch_loss["train"]))

            if self.valset != None:

                self.model.eval()
                for img, label in tqdm(self.val_dl):

                    with torch.cuda.amp.autocast():

                        pred = self.model(img)
                        loss = self.loss_fn(pred, label)

                        epoch_loss["val"].append(loss)

                        if self.mode == "classification":
                            a = accuracy(pred, label)
                            epoch_acc["val"].append(a)

                        elif self.mode == "detection":
                            a = accuracy(pred[1:5], label[1:5])
                            epoch_acc["val"].append(a)

                losses["val"].append(sum(epoch_loss["val"]) / len(epoch_loss["val"]))

                if flag:

                    acc["val"].append(sum(epoch_acc["val"]) / len(epoch_acc["val"]))
                    acc["train"].append(
                        sum(epoch_acc["train"]) / len(epoch_acc["train"])
                    )

                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]} -- Train acc: {acc['train'][-1]}% -- Val Loss: {losses['val'][-1]} -- Val acc: {acc['val'][-1]}%"
                    )
                else:
                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]} -- Val Loss: {losses['val'][-1]}"
                    )

            else:

                if flag:
                    acc["train"].append(
                        sum(epoch_acc["train"]) / len(epoch_acc["train"])
                    )

                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]} -- Train acc: {acc['train'][-1]}%"
                    )
                else:
                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]}"
                    )

            torch.save(self.model, f"{self.model_save_path}\\model")

        if flag:
            return losses, acc

        else:
            return losses

    def test_sample(self, image, label=None):
        """
        Used to test the model on one image.

        Returns the prediction.
        """

        pred = self.model(image)

        if label != None:
            loss = self.loss_fn(label, pred).detach()
            return pred, loss

        return pred

    def evaluate(self, test_path):
        """
        Used to evaluate the model on the test dataset. 

        Returns the losses. 
        """

        test_dl = get_dataloader(
            ImageDataset(test_path, self.mode, device), self.b_size, False
        )
        losses = []

        for img, label in test_dl:
            pred = self.model(img)
            loss = self.loss_fn(label, pred).detach()
            losses.append(loss)

        return sum(losses) / len(losses)

# test pull request. 