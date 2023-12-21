import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

class Learner(pl.LightningModule):
    def __init__(self, net, criterion):
        super().__init__()
        self.net = net
        self.criterion = criterion


    def forward(self, batch):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        x, y = batch # image, label

        y_pred = self.net(x)

        return y_pred, y

    def step(self, batch, mode, ):
        # Forward pass
        y_pred, y = self.forward(batch)

        # Loss
        loss = self.criterion(y_pred, y)

        # Metrics
        acc = self.get_accuracy(y_pred, y)
        self.log(f"{mode}_acc", acc, prog_bar=True)#, on_step=True, on_epoch=True)

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)#, on_step=True, on_epoch=True)

        return loss, batch, y_pred

    def training_step(self, batch, batch_idx):
        loss, batch, y_pred = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, batch, y_pred = self.step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx):
        loss, batch, y_pred = self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def get_accuracy(self, y_pred, y):
        return torch.sum(y_pred.argmax(dim=1) == y).item() / len(y)