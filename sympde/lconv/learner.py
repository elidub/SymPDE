import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from lconv.generator import Generator

class LconvLearner(pl.LightningModule):
    def __init__(self, net, criterion, r, dim):
        super().__init__()
        self.net = net
        self.criterion = criterion

        gen = Generator(dim = dim)
        self.L_gt = gen.xLy - gen.yLx
        self.r = r

    def forward(self, batch):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        x, y = batch # image, image_rot

        y_pred = self.net(x)
        y_pred = y_pred.squeeze(2)

        return y_pred, y

    def step(self, batch, mode="train"):
        # Forward pass
        y_pred, y = self.forward(batch)

        # Loss
        loss = self.criterion(y_pred, y)

        # Metrics
        L_pred = self.get_Lpred(batch)
        mse = torch.mean((torch.tensor(L_pred) - self.L_gt)**2)
        self.log(f"{mode}_Lmse", mse, prog_bar=True, on_step=False, on_epoch=True)

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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

        if batch_idx == 0:
            self.log_fig(batch, y_pred, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def get_Lpred(self, batch):
        L = self.net.layers[1].L.detach().cpu().squeeze(0).numpy()
        Wi = self.net.layers[1].Wi.detach().cpu().squeeze(0).numpy()

        ang = np.pi / self.r if self.r > 0 else 1
        L_pred = L * Wi / ang
        return L_pred

    
    def log_fig(self, batch, preds, mode = None, sample_id = 0):

        L_pred = self.get_Lpred(batch)

        fig, axs = plt.subplots(1, 2, figsize=(6, 5), tight_layout=True)
        cmap = 'seismic'
        img = axs[0].imshow(L_pred, cmap=cmap)
        fig.colorbar(img, ax=axs[0])
        axs[0].set_title(r'Ground truth $L_\theta=x\partial_y-y\partial_x$')

        img = axs[1].imshow(self.L_gt, cmap=cmap)
        fig.colorbar(img, ax=axs[1])
        axs[1].set_title(r'Learned $L$ (fixed $\theta=\pi/%d$)'%(self.r))

        if mode is not None:
            self.logger.experiment.add_figure(f'{mode}_plot', fig, self.current_epoch)
        plt.close(fig)


        images, images_rot = batch
        n_examples = 5
        x = images[:n_examples].detach().cpu().squeeze().numpy()
        x_rot = images_rot[:n_examples].detach().cpu().squeeze().numpy()
        y = preds[:n_examples].detach().cpu().squeeze().numpy()

        fig, axs = plt.subplots(3, n_examples, figsize=(10, 5), sharex=True, sharey=True, tight_layout=True)
        for i in range(n_examples):
            axs[0, i].imshow(x[i], cmap = 'gray')
            axs[1, i].imshow(x_rot[i], cmap = 'gray')
            axs[2, i].imshow(y[i], cmap = 'gray')
            axs[0, 0].set_ylabel('Original')
            axs[1, 0].set_ylabel('Rotated')
            axs[2, 0].set_ylabel('Predicted')

            for ax in axs:
                ax[i].set_xticks([])
                ax[i].set_yticks([])

        if mode is not None:
            self.logger.experiment.add_figure(f'{mode}_images', fig, self.current_epoch)

        plt.close(fig)



