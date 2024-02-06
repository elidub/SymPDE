import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb

import sklearn.metrics as skm

from data.transforms import Transform
from misc.utils import NumpyUtils

class BaseLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

        self.test_step_outs = []
        self.forward_keys = []


    def forward(self, batch):
        raise NotImplementedError
    
    def test_logs_method(self):
        return {}
    
    def step(self, batch, mode):
        out = self.forward(batch)
        loss = self.criterion(*out)

        # Log Metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, batch, out
    
    def training_step(self, batch, batch_idx=0):
        loss, batch, _ = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, _ = self.step(batch, "val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, out = self.step(batch, "test")
        self.test_step_outs.append(out)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_test_end(self):

        if not self.trainer.logger:
            print("No logger, skipping logging")
            return
        
        run_id = self.trainer.logger.experiment.id

        save =  NumpyUtils(dir=self.trainer.logger.experiment.dir).save

        pred_outs = zip(*self.test_step_outs)
        pred_outs = [torch.cat(pred_out).cpu().numpy() for pred_out in pred_outs]
        for forward_key, pred_out in zip(self.forward_keys, pred_outs):
            save(forward_key, pred_out)

        # TODO: Automatize this such taht it doesn't happen all the time
        # y_preds, y_trues = pred_outs
        # self.on_test_end_extra(y_preds, y_trues)

        for key, value in self.test_logs_method().items():
            print(f'Logging {key}')
            store_dir = os.path.join(self.trainer.log_dir, 'store', key)
            os.makedirs(store_dir, exist_ok=True)
            np.save(os.path.join(store_dir, f'{run_id}.npy'), value.cpu().numpy())
            
    
class PredictionLearner(BaseLearner):
    def __init__(self, net, criterion, lr):
        super().__init__(net, criterion, lr)

        self.forward_keys = ['y_true', 'y_pred']

    def forward(self, batch):

        x, y_true, _ = batch

        y_pred = self.net(x.unsqueeze(1)).squeeze(1).squeeze(1)

        return y_pred, y_true
    
    def on_test_end_extra(self, y_preds, y_trues):
        y_hats = np.argmax(y_preds, axis = 1)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
        disp = skm.ConfusionMatrixDisplay.from_predictions(y_trues, y_hats)
        disp.plot(ax=ax, colorbar=False)
        plt.close()
        wandb.log({'confusion_matrix': wandb.Image(fig)})
    
class TransformationLearner(BaseLearner, Transform):
    def __init__(self, net, criterion, lr, grid_size, transform_kwargs):
        BaseLearner.__init__(self, net, criterion, lr)
        Transform.__init__(self, grid_size, **transform_kwargs)

    def forward(self, batch):

        x, y_, centers = batch

        batch_size = len(x)
        eps = torch.randn((4,))

        # Reset the weights and biases as training P should not be dependent on the weight initailization
        if self.net.train_P:
            self.net.reset_parameters(batch_size=batch_size)

        # Route a: Forward pass, transformation
        x_a = x
        out_a = self.net(x_a, batch_size=batch_size)
        out_a_prime, centers_a = self.transform(out_a, centers, eps)

        # Route b: Transformation, forward pass
        x_b = x
        x_b_prime, centers_b = self.transform(x_b, centers, eps)
        out_b_prime = self.net(x_b_prime, batch_size=batch_size)

        assert (centers_a == centers_b).all()
        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        return out_a_prime, out_b_prime
    
    def test_logs_method(self):
        return {'P': self.net.P}
