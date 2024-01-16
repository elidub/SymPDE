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
    
class FlatLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

    def space_translation(self, x, eps):
        batch, width = x.shape
        shift = (eps * width).to(torch.int)

        x_prime = torch.stack([
            torch.roll(x_i, shifts=int(shift_i), dims=0)
        for x_i, shift_i in zip(x, shift)])

        return x_prime
    
    def aug_dev(self, x, eps):
        # print(x.shape)
        x_prime = x
        # x_prime = x * torch.exp(eps - 0.5) 
        # x_prime = x + (torch.exp(eps - 0.5) - 1)
        # x_prime += torch.tensor([0., 0.5, 0.5, 0.])
        # x_prime = x * 0
        return x_prime


    def forward(self, batch):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        x, y_, eps = batch

        transf = self.space_translation
        # transf = self.aug_dev

        if self.net.transform_type == 'train_Pw':
            self.net.reset_parameters()

        x_a = x
        out_a = self.net(x_a.unsqueeze(1)).squeeze(1)
        out_a_prime = transf(out_a, eps)

        x_b = x
        x_b_prime = transf(x_b, eps)
        out_b_prime = self.net(x_b_prime.unsqueeze(1)).squeeze(1)


        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        # print('x_a', x_a, sep = '\n')
        # print('out_a', out_a, sep = '\n')
        # print('out_a_prime', out_a_prime, sep = '\n')

        # print('x_b', x_b, sep = '\n')
        # print('x_b_prime', x_b_prime, sep = '\n')
        # print('out_b_prime', out_b_prime, sep = '\n')

        return out_a_prime, out_b_prime

    def step(self, batch, mode, ):
        # Forward pass
        o_a_prime, o_b_prime = self.forward(batch)

        # Loss
        loss = self.criterion(o_a_prime, o_b_prime)

        # Metrics
        # more metrics to come ...

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)#, on_step=True, on_epoch=True)

        return loss, batch, (o_a_prime, o_b_prime)

    def training_step(self, batch, batch_idx=0):
        loss, batch, _ = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, _ = self.step(batch, "val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, _ = self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class FlatYLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

    def forward(self, batch):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        x, y_true, eps_ = batch

        y_pred = self.net(x.unsqueeze(1)).squeeze(1).squeeze(1)

        return y_true, y_pred

    def step(self, batch, mode, ):
        # Forward pass
        y_true, y_pred = self.forward(batch)

        # Loss
        loss = self.criterion(y_true, y_pred)

        # Metrics
        # more metrics to come ...

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)#, on_step=True, on_epoch=True)

        return loss, batch, (y_true, y_pred)

    def training_step(self, batch, batch_idx=0):
        loss, batch, _ = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, _ = self.step(batch, "val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, _ = self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer