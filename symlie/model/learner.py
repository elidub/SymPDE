import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
import torchvision

import sklearn.metrics as skm

from data.transforms import Transform
from misc.utils import NumpyUtils
from model.networks.linear import LinearP
from model.networks.implicit import LinearImplicit

class BaseLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

        self.test_step_outs = []

        if type(criterion) == list:
            if len(criterion) in [2, 8]:
                self.criterion_alt = True
            else:
                raise NotImplementedError(f"Criterion {criterion} not implemented")
        else:
            self.criterion_alt = False


    def forward(self, batch):
        raise NotImplementedError
    
    def log_test_results(self):
        pass

    def step(self, batch, mode):

        if self.criterion_alt:
            loss, batch, out = self.step_alt(batch, mode)
        else:
            out = self.forward(batch)
            loss = self.criterion(*out)

            # Log Metrics
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, batch, out
    
    def step_alt(self, batch, mode):
        out = self.forward(batch)


        out_terms = out
        loss_terms = self.criterion
        log_terms = ['loss_o', 'loss_dg', 'loss_dx', 'loss_do', 'loss_do_a', 'loss_do_b', 'loss_do_a_mmd', 'loss_do_b_mmd']
        assert len(out_terms) == len(loss_terms) == len(log_terms)

        loss = 0
        for out, (lossweight, criterion), log_term in zip(out_terms, loss_terms, log_terms):
            loss_term = criterion(*out)
            loss += lossweight*loss_term
            self.log(f"{mode}_{log_term}", loss_term, prog_bar=True, on_step=False, on_epoch=True)

        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # out_o, out_dg = out
        # (lossweight_o, criterion_o), (lossweight_dg, criterion_dg) = self.criterion
        
        # loss_o = criterion_o(*out_o)
        # loss_dg = criterion_dg(*out_dg)
        # loss = lossweight_o*loss_o + lossweight_dg*loss_dg

        # # Log Metrics
        # self.log(f"{mode}_loss_o", loss_o, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss_dg", loss_dg, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        ### Vanilla tilde ####
        # out_o, out_do_a, out_do_b = out
        # (lossweight_o, criterion_o), (lossweight_do_a, criterion_do_a), (lossweight_do_b, criterion_do_b) = self.criterion
        
        # loss_o = criterion_o(*out_o)
        # loss_do_a = criterion_do_a(*out_do_a)
        # loss_do_b = criterion_do_b(*out_do_b)
        # loss = lossweight_o*loss_o + lossweight_do_a*loss_do_a + lossweight_do_b*loss_do_b

        # # Log Metrics
        # self.log(f"{mode}_loss_o", loss_o, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss_do_a", loss_do_a, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss_do_b", loss_do_b, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        #### Four losses ####
        # out_o, out_dg, out_dx, out_do = out
        # (lossweight_o, criterion_o), (lossweight_dg, criterion_dg), (lossweight_dx, criterion_dx), (lossweight_do, criterion_do) = self.criterion
        
        # loss_o = criterion_o(*out_o)
        # loss_dg = criterion_dg(*out_dg)
        # loss_dx = criterion_dx(*out_dx)
        # loss_do = criterion_do(*out_do)
        # loss = lossweight_o*loss_o + lossweight_dg*loss_dg + lossweight_dx*loss_dx + lossweight_do*loss_do

        # # Log Metrics
        # self.log(f"{mode}_loss_o", loss_o, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss_dg", loss_dg, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss_dx", loss_dx, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss_do", loss_do, prog_bar=True, on_step=False, on_epoch=True)
        # self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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

        print('Print parameters in configure_optimizers')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_test_end(self):

        # if not self.trainer.logger:
        #     print("No logger, skipping logging")
        #     return
        
        self.log_test_results()
        
class TransformationLearner(BaseLearner, Transform):
    def __init__(self, net, criterion, lr, grid_size, transform_kwargs):
        BaseLearner.__init__(self, net, criterion, lr)
        Transform.__init__(self, grid_size, **transform_kwargs)

        # size = torch.prod(torch.tensor(grid_size)).item()
        # self.generator = self.init_generator_learner(size)

    # def init_generator_learner(self, size):
    #     print(f'Initializing generator with size {size}')
    #     mlp = torchvision.ops.MLP(
    #         in_channels = size + len(self.eps_mult),
    #         hidden_channels = [size, size],
    #     )
    #     return mlp
    
    def transform_stripped(self, x, size, eps):
        shift = (eps * size).int().item()
        return torch.roll(x, shift, 1)

    def forward(self, batch):

        x, y_, centers_ = batch

        batch_size = len(x)
        batch_size = None
        eps = torch.randn((4,))

        # Reset the weights and biases as training P should not be dependent on the weight initailization
        if self.net.train_P:
            self.net.reset_parameters(batch_size=batch_size)

        # Route a: Forward pass, transformation
        x_a = x
        out_a = self.net(x_a, batch_size)
        # out_a_prime = self.transform_stripped(out_a, size, eps)
        out_a_prime, _ = self.transform(out_a, centers_, eps)

        # Route b: Transformation, forward pass
        x_b = x
        # x_b_prime = self.transform_stripped(x_b, size, eps)
        x_b_prime, _ = self.transform(x_b, centers_, eps)
        out_b_prime = self.net(x_b_prime, batch_size)

        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        return (out_a_prime, out_b_prime)


    def forward_old(self, batch):

        x, y_, centers = batch

        batch_size = len(x)
        batch_size = None
        eps = torch.randn((4,))

        # Reset the weights and biases as training P should not be dependent on the weight initailization
        if self.net.train_P:
            self.net.reset_parameters(batch_size=batch_size)

        # Route a: Forward pass, transformation
        x_a = x
        out_a = self.net(x_a, batch_size=batch_size)
        out_a_prime, _ = self.transform(out_a, centers, eps)

        # Route b: Transformation, forward pass
        x_b = x
        x_b_prime, _ = self.transform(x_b, centers, eps)
        out_b_prime = self.net(x_b_prime, batch_size=batch_size)

        # Vanilla tilde
        weight = self.net.weight
        out_a_tilde = torch.einsum('bi,boi->bo', x_a, weight)
        out_a_prime_tilde, _ = self.transform(out_a_tilde, centers, eps)

        out_b_prime_tilde = torch.einsum('bi,boi->bo', x_b_prime, weight)

        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        criterion_alt = True
        if criterion_alt:

            # placeholder = torch.zeros_like(out_a)
            # return (placeholder, placeholder), (placeholder, placeholder), (placeholder, placeholder), (placeholder, placeholder), (out_a_prime, out_a_prime_tilde), (placeholder, placeholder)

            eps_multed = eps * self.eps_mult
            eps_multed = eps_multed.repeat(batch_size, 1).to(x_a.device)

            phi_x_a   = self.generator(torch.cat([x_a, eps_multed], dim=1)) 
            phi_out_a = self.generator(torch.cat([out_a, eps_multed], dim=1))

            dg_x = phi_x_a - x_b_prime
            dg_out = phi_out_a - out_b_prime

            return (out_a_prime, out_b_prime), (dg_x, dg_out), (phi_x_a, x_b_prime), (phi_out_a, out_b_prime), (out_a_prime, out_a_prime_tilde), (out_b_prime, out_b_prime_tilde), (out_a_prime, out_a_prime_tilde), (out_b_prime, out_b_prime_tilde)

            dw_a = x_a - out_a
            dw_a_tilde = x_a - out_a_tilde

            dw_b_prime = x_b_prime - out_b_prime
            dw_b_prime_tilde = x_b_prime - out_b_prime_tilde

            return (out_a_prime, out_b_prime), (dg_x, dg_out), (phi_x_a, x_b_prime), (phi_out_a, out_b_prime), (dw_a, dw_a_tilde), (dw_b_prime, dw_b_prime_tilde)

        return (out_a_prime, out_b_prime)
    
    def log_test_results(self):
        # run_id = self.trainer.logger.experiment.id
        # log_dir = self.trainer.log_dir

        run_id = 'temp_runid'
        log_dir = '../logs'

        if hasattr(self.net, 'svd'):
            if self.net.svd: 
                P = self.net.U @ torch.diag(self.net.S) @ self.net.V
                logging_objects = {'P' : P, 'U': self.net.U, 'S': self.net.S, 'V': self.net.V}
            else:
                logging_objects = {'P': self.net.P}
            save_format = 'numpy'
        elif hasattr(self.net, 'implicit_P'):
            print('Logging implicit_P!!')
            logging_objects = {'implicit_P': self.net.implicit_P.state_dict()}
            print(self.net.implicit_P.state_dict())
            save_format = 'state_dict'
        # elif hasattr(self.net, 'layers'):
        #     print('Logging layers!!')
        #     logging_objects = {'implicit_P': self.net.layers.state_dict()}
        #     print(self.net.layers.state_dict())
        #     save_format = 'state_dict'
        else:
            raise NotImplementedError(f"Logging not implemented for {self.net}")
        
        
        plt.figure(figsize=(3,3))
        size = 7
        self.net.reset_parameters()
        plt.imshow( self.net(torch.rand(1, size), return_weight = True).detach().cpu().numpy() )
        plt.savefig('../notebooks_symlie/mlp_weight.png')
        plt.close()

        for key, value in logging_objects.items():
            print(f'Logging {key}')
            store_dir = os.path.join(log_dir, 'store', key)
            os.makedirs(store_dir, exist_ok=True)
            if save_format == 'numpy':
                np.save(os.path.join(store_dir, f'{run_id}.npy'), value.cpu().numpy())
            elif save_format == 'state_dict':
                torch.save(value, os.path.join(store_dir, f'{run_id}.pt'))

    
class PredictionLearner(BaseLearner):
    def __init__(self, net, criterion, lr, task):
        super().__init__(net, criterion, lr)
        self.task = task

    def forward(self, batch):

        x, y_true, _ = batch

        y_pred = self.net(x.unsqueeze(1)).squeeze(1).squeeze(1)

        y_true = y_true.squeeze(1)

        return y_pred, y_true
    
    def log_test_results(self):
        pred_outs = zip(*self.test_step_outs)
        pred_outs = [torch.cat(pred_out).cpu().numpy() for pred_out in pred_outs]

        tasks = {'classification': self._log_classification, 'regression': self._log_regression}
        tasks[self.task](*pred_outs)

    
    def _log_classification(self, y_preds, y_trues):
        y_hats = np.argmax(y_preds, axis = 1)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
        disp = skm.ConfusionMatrixDisplay.from_predictions(y_trues, y_hats)
        disp.plot(ax=ax, colorbar=False)
        plt.close()
        wandb.log({'confusion_matrix': wandb.Image(fig)})

    def _log_regression(self, y_preds, y_trues):

        print(y_preds.shape, y_trues.shape)

        if len(y_preds.shape) == 1:
            y_preds = y_preds.reshape(-1, 1)
            y_trues = y_trues.reshape(-1, 1)

        n_cols = len(y_preds.T)
        fig, axs = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

        if len(y_preds.T) == 1: axs = [axs]

        for ax, y_trues_i, y_preds_i in zip(axs, y_trues.T, y_preds.T):

            l_min, l_max = np.min(y_trues_i)*0.9, np.max(y_trues_i)*1.1
            ax.plot([l_min, l_max], [l_min, l_max], 'k--')
            ax.plot(y_trues_i, y_preds_i, '.', alpha=0.5)
        fig.supxlabel('True')
        fig.supylabel('Predicted')
        plt.close()
        wandb.log({'regression_results': wandb.Image(fig)})

        print('Logged regression results')