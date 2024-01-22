import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from data.transforms import Transform

class BaseLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

    def forward(self, batch):
        raise NotImplementedError
    
    def step(self, batch, mode):
        out = self.forward(batch)
        loss = self.criterion(*out)

        # Log Metrics
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss, batch, out
    
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
    
class PredictionLearner(BaseLearner):
    def __init__(self, net, criterion, lr):
        super().__init__(net, criterion, lr)

    def forward(self, batch):

        x, y_true, _ = batch

        y_pred = self.net(x.unsqueeze(1)).squeeze(1).squeeze(1)

        return y_true, y_pred


class TransformationLearner(BaseLearner):
    def __init__(self, net, criterion, lr):
        super().__init__(net, criterion, lr)

    def space_translation(self, x, eps):
        """
        TODO: vectorize transformation and move to dataset
        """
        _, width = x.shape
        shift = (eps * width).to(torch.int)

        x_prime = torch.stack([
            torch.roll(x_i, shifts=int(shift_i), dims=0)
        for x_i, shift_i in zip(x, shift)])

        return x_prime
    
    def forward(self, batch):

        x, y_, eps = batch

        # Hardcoded transformation # TODO: Automatize this
        transf = self.space_translation

        # Reset the weights and biases as training P should not be dependent on the weight initailization
        if self.net.train_P:
            self.net.reset_parameters()

        # Route a: Forward pass, transformation
        x_a = x
        out_a = self.net(x_a.unsqueeze(1)).squeeze(1)
        out_a_prime = transf(out_a, eps)

        # Route b: Transformation, forward pass
        x_b = x
        x_b_prime = transf(x_b, eps)
        out_b_prime = self.net(x_b_prime.unsqueeze(1)).squeeze(1)

        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        return out_a_prime, out_b_prime
    

class FlowerTransformationLearner(BaseLearner, Transform):
    def __init__(self, net, criterion, lr, transform_kwargs):
        BaseLearner.__init__(self, net, criterion, lr)
        Transform.__init__(self, only_flip=transform_kwargs['only_flip'])
        self.eps_mult = transform_kwargs['eps_mult']

    def forward(self, batch):

        x, y_, eps_, centers = batch

        batch_size = len(x)
        eps = torch.randn((4,))
        eps = eps * torch.tensor(self.eps_mult)

        # Reset the weights and biases as training P should not be dependent on the weight initailization
        if self.net.train_P:
            self.net.reset_parameters()

        # Route a: Forward pass, transformation
        x_a = x
        out_a = self.net(x_a.unsqueeze(1)).squeeze(1)
        out_a_prime, centers_a = self.transform(out_a, centers, eps)

        # Route b: Transformation, forward pass
        x_b = x
        x_b_prime, centers_b = self.transform(x_b, centers, eps)
        out_b_prime = self.net(x_b_prime.unsqueeze(1)).squeeze(1)

        assert (centers_a == centers_b).all()
        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        return out_a_prime, out_b_prime
    
