import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from viz.plot_pde_data import plot_pred

class Learner(pl.LightningModule):
    def __init__(self, net, criterion, optimizer, lr, scheduler):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler

        self.x_start = 0
        self.y_start = self.x_end = self.net.time_history
        self.y_end = self.y_start + self.net.time_future

    def forward(self, batch):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        us, dx, dt = batch

        # [batch, time, space] -> [batch, space, time]
        us = us.permute(0, 2, 1) 

        # Select the time history and future for input and target
        x = us[:, :, :self.x_end]        
        y = us[:, :, self.y_start:self.y_end] 

        # Pass the time history through the network
        y_pred = self.net(x, dx, dt)

        # [batch, space, time] -> [batch, time, space]
        y_pred = y_pred.permute(0, 2, 1)
        y      = y.permute(0, 2, 1)

        return y_pred, y

    def step(self, batch, mode="train"):
        # Forward pass
        y_pred, y = self.forward(batch)

        # Loss
        loss = self.criterion(y_pred, y)

        # Metrics
        # Additional metrics can be calculated here

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss, batch, y_pred

    def training_step(self, batch, batch_idx):
        loss, batch, y_pred = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch, y_pred = self.step(batch, "val")

        if batch_idx == 0:
            self.log_fig(batch, y_pred, "val")

    def test_step(self, batch, batch_idx):
        loss, batch, y_pred = self.step(batch, "test")
        # self.us_batches.append(batch)
        # self.y_preds_batches.append(y_pred)

        if batch_idx == 0:
            self.log_fig(batch, y_pred, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer, milestones=[0, 5, 10, 15], gamma=0.4)
        return [optimizer], [scheduler]
    
    
    # def on_test_epoch_start(self):
    #     # Initialize lists to store inputs and predictions
    #     self.us_batches = []
    #     self.y_preds_batches = []

    # def on_test_epoch_end(self):
    #     return self.us_batches, self.y_preds_batches
    
    def log_fig(self, batch, preds, mode = None, sample_id = 0):
        x_start, x_end, y_start, y_end  = self.x_start, self.x_end, self.y_start, self.y_end

        us, dx, dt = batch
        u = us[sample_id].cpu().numpy()
        pred = preds[sample_id].cpu().numpy()

        input = u[x_start:y_end]
        output = np.full_like(input, np.nan)
        output[y_start:y_end] = pred

        fig = plot_pred(input, output, dx, dt, x_start, x_end, y_start, y_end)
        if mode is not None:
            self.logger.experiment.add_figure(f'{mode}_plot', fig, self.current_epoch)
        plt.close(fig)
        return fig

