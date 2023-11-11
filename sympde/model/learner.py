import torch
import torch.nn as nn
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    def __init__(self, net, criterion):
        super().__init__()
        self.net = net
        self.criterion = criterion

    def forward(self, batch):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        y_start = x_end = self.net.time_history
        y_end = self.net.time_history+self.net.time_future

        us, dx, dt = batch

        # [batch, time, space] -> [batch, space, time]
        us = us.permute(0, 2, 1) 

        # Select the time history and future for input and target
        x = us[:, :, :x_end]        
        y = us[:, :, y_start:y_end] 

        # Pass the time history through the network
        y_hat = self.net(x, dx, dt)

        # [batch, space, time] -> [batch, time, space]
        y_hat = y_hat.permute(0, 2, 1)
        y     = y.permute(0, 2, 1)

        return y_hat, y

    def step(self, batch, mode="train"):
        # Forward pass
        y_hat, y = self.forward(batch)

        # Loss
        loss = self.criterion(y_hat, y)

        # Metrics
        # Additional metrics can be calculated here

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_end(self):
        pass
