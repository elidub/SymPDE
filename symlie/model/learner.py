import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
import torchvision
import torch.nn.functional as F

import sklearn.metrics as skm

from data.transforms import Transform, TransformRefactored
from misc.utils import NumpyUtils
from model.networks.linear import LinearP
from model.networks.implicit import LinearImplicit

from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

torch.autograd.set_detect_anomaly(True)

OLD_COMMIT = True # edeb8f0 (https://github.com/elidub/SymPDE/blob/edeb8f01e039cbc1a0b1a4926df1dd72dc60b736/symlie/model/learner.py)

class BaseLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr, optimizer_setting, **kwargs):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

        self.test_step_outs = []

        self.criterion_alt = True
        self.optimizer_setting = optimizer_setting

        return


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


    # def step(self, batch, mode):
        # out_y = self.forward_vanilla(batch)
        # out_ab_primes = self.forward_implicit(batch)

    def step(self, batch, mode):

        if self.criterion_alt:
            loss, batch, out = self.step_alt(batch, mode)
            # loss, batch, out = self.step_alt_old(batch, mode)
        else:
            out = self.forward(batch)
            
            try:
                _, criterion = self.criterion[0]
            except:
                criterion = self.criterion

            loss = criterion(*out)

            # Log Metrics
            self.log(f"{mode}_loss_y", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, batch, out

    def step_alt(self, batch, mode):

        if OLD_COMMIT:
            out = self.forward(batch)

            out_terms = out
            loss_terms = self.criterion
            log_terms = ['loss_y'] + [f'loss_o{i}' for i in range(len(loss_terms)-1)]
            assert len(out_terms) == len(loss_terms) == len(log_terms), f"Length mismatch: {len(out_terms)}, {len(loss_terms)}, {len(log_terms)}"

            loss = 0
            for out, (lossweight, criterion), log_term in zip(out_terms, loss_terms, log_terms):
                loss_term = criterion(*out)
                self.log(f"{mode}_{log_term}", loss_term, prog_bar=True, on_step=False, on_epoch=True)
                loss += lossweight*loss_term

            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

            out = out_terms[0] # Only select the prediction of y

            return loss, batch, out




        if self.optimizer_setting == 'solo':

            assert len(self.criterion) == 1, self.criterion
            lossweight_y, criterion_y = self.criterion[0]

            out_y = self.forward_vanilla(batch)
            loss_y = criterion_y(*out_y)

            self.log(f"{mode}_loss_y", loss_y, prog_bar=True, on_step=False, on_epoch=True)

            return loss_y, batch, out_y

        loss = 0

        opt_vanilla, opt_implicit = self.optimizers()

        # (lossweight_y, criterion_y), (lossweight_o, criterion_o), _ = self.criterion
        (lossweight_y, criterion_y), (lossweight_o, criterion_o) = self.criterion

        self.net.reset_parameters_vanilla()


        ## Vanilla, y ##

        self.set_grad(grad_false = self.net.implicit_layers.parameters(), grad_true = self.net.vanilla_layers.parameters())
        out_y = self.forward_vanilla(batch)
        loss_y = criterion_y(*out_y)

        if lossweight_y > 0. and mode == 'train' and not self.automatic_optimization: 
            opt_vanilla.zero_grad()
            self.manual_backward(loss_y, retain_graph=True)
            opt_vanilla.step()

        loss += lossweight_y*loss_y
        self.log(f"{mode}_loss_y", loss_y, prog_bar=True, on_step=False, on_epoch=True)

        ## Implicit, o ##

        self.set_grad(grad_true = self.net.implicit_layers.parameters(), grad_false = self.net.vanilla_layers.parameters())

        out_ab_primes = self.forward_implicit(batch)

        loss_os = 0
        for i, (out_a_prime, out_b_prime) in enumerate(out_ab_primes):
            loss_o = criterion_o(out_a_prime, out_b_prime)
            loss += lossweight_o*loss_o
            loss_os += loss_o
            self.log(f"{mode}_loss_o{i}", loss_o, prog_bar=True, on_step=False, on_epoch=True)

        if lossweight_o > 0. and mode == 'train' and not self.automatic_optimization:
            opt_implicit.zero_grad()
            self.manual_backward(loss_os)
            opt_implicit.step()


        return loss, batch, out_y


    def step_alt_old(self, batch, mode):
        out = self.forward(batch)


        out_terms = out
        loss_terms = self.criterion
        # log_terms = ['loss_o', 'loss_dg', 'loss_dx', 'loss_do', 'loss_do_a', 'loss_do_b', 'loss_do_a_mmd', 'loss_do_b_mmd']
        log_terms = ['loss_y'] + [f'loss_o{i}' for i in range(len(loss_terms)-1)]
        assert len(out_terms) == len(loss_terms) == len(log_terms), f"Length mismatch: {len(out_terms)}, {len(loss_terms)}, {len(log_terms)}"


        loss = 0
        losses = []
        for i, (out, (lossweight, criterion), log_term) in enumerate(zip(out_terms, loss_terms, log_terms)):
            loss_term = criterion(*out)
            losses.append(loss_term)
            self.log(f"{mode}_{log_term}", loss_term, prog_bar=True, on_step=False, on_epoch=True)
            loss += lossweight*loss_term
            # loss += self.adapt_weights[i]*loss_term


        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        out = out_terms[0] # Only select the prediction of y

        if mode == 'train' and not self.automatic_optimization:

            with torch.autograd.set_detect_anomaly(True):

                (lossweight_y, _), (lossweight_o, _), _ = loss_terms

                opt_vanilla, opt_implicit = self.optimizers()

                loss_vanilla = losses[0]
                loss_implicit = sum(losses[1:])

                if lossweight_o > 0.:
                    self.set_grad(grad_true = self.net.implicit_layers.parameters(), grad_false = self.net.vanilla_layers.parameters())
                    opt_implicit.zero_grad()
                    self.manual_backward(loss_implicit, retain_graph=False)
                    opt_implicit.step()

                if lossweight_y > 0.:
                    self.set_grad(grad_false = self.net.implicit_layers.parameters(), grad_true = self.net.vanilla_layers.parameters())
                    opt_vanilla.zero_grad()
                    self.manual_backward(loss_vanilla, retain_graph=False)
                    opt_vanilla.step()

        return loss, batch, out
    
    
    def training_step(self, batch, batch_idx=0):
        loss, batch, _ = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, _ = self.step(batch, "val")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        loss, batch, out = self.step(batch, "test")
        self.test_step_outs.append(out)

    def configure_optimizers_solo(self):

        print('Print parameters in configure_optimizers')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def configure_optimizers_multi(self):

        self.automatic_optimization = False

        print('Vanilla layers')
        for name, param in self.net.vanilla_layers.named_parameters():
            print(name, param.requires_grad)
        print()

        print('Implicit layers')
        for name, param in self.net.implicit_layers.named_parameters():
            print(name, param.requires_grad)
        print()

        opt_vanilla  = torch.optim.Adam(self.net.vanilla_layers.parameters(), lr=self.lr)
        opt_implicit = torch.optim.Adam(self.net.implicit_layers.parameters(), lr=self.lr)
        return opt_vanilla, opt_implicit
        
    def configure_optimizers(self):

        if self.optimizer_setting == 'solo':
            print('Configuring solo optimizer')
            return self.configure_optimizers_solo()
        elif self.optimizer_setting == 'multi':
            print('Configuring multi optimizer')
            return self.configure_optimizers_multi()

    def on_test_end(self):

        if not self.trainer.logger:
            print("No logger, skipping logging")
            return
        
        self.log_test_results()

class TransformationBlock(TransformRefactored):
    def __init__(self, transform_kwargs, **kwargs):
        TransformRefactored.__init__(self, eps_mult = transform_kwargs['eps_mult'])
        self.rng_a, self.rng_b = torch.Generator(), torch.Generator()
        print('Init rng')

    def forward_transformation(self, batch_size, x_in, shape, weight, bias):
        self.rng_b.set_state(self.rng_a.get_state())
        assert len(shape) == 2, shape
        shape = {'a':shape[0], 'b':shape[1]}

        x_a = x_b = torch.randn((batch_size, np.prod(shape['b'])), device = weight.device)
        # x_a = x_b = x_in

        # Route a: Forward pass, transformation
        out_a = F.linear(x_a, weight, bias)
        out_a_prime = self.transform(out_a, self.rng_a, shape=shape['a'])

        # Route b: Transformation, forward pass
        x_b_prime = self.transform(x_b, self.rng_b, shape=shape['b'])
        out_b_prime = F.linear(x_b_prime, weight, bias)

        assert out_a_prime.shape == out_b_prime.shape
        

        return (out_a_prime, out_b_prime)
        


class CombiLearner(BaseLearner, TransformationBlock):
    def __init__(self, net, criterion, lr, grid_sizes, transform_kwargs, optimizer_setting):
        kwargs = {'net': net, 'criterion': criterion, 'lr': lr, 'transform_kwargs': transform_kwargs, 'optimizer_setting': optimizer_setting}
        super().__init__(**kwargs)
        BaseLearner.__init__(self, net, criterion, lr, optimizer_setting)
        TransformationBlock.__init__(self, transform_kwargs)
        self.grid_sizes = grid_sizes


    def set_grad(self, grad_true, grad_false):
        try:
            for param in grad_true:
                param.requires_grad = True
            for param in grad_false:
                param.requires_grad = False
        except:
            pass

    def forward(self, batch):

        if OLD_COMMIT:
            out_y = self.forward_vanilla(batch)
            out_ab_primes = self.forward_implicit(batch)
            return out_y, *out_ab_primes

        # self.net.reset_parameters_vanilla()


        x, y_true, _ = batch
        batch_size = len(x)

        # for param in self.net.parameters():
        #     param.requires_grad = True

        # disable grad of implicit layers but keep grad of vanilla layers
        self.set_grad(grad_false = self.net.implicit_layers.parameters(), grad_true = self.net.vanilla_layers.parameters())
        
        y_pred = self.net(x)
        out_y = (y_pred.squeeze(1), y_true.squeeze(1))

        # return (y_pred.squeeze(1), y_true.squeeze(1)),

        # out_ab_primes = []
        # for grid_size, weight, layer in zip(self.grid_sizes, self.net.weights, self.net.layers):
        #     weight_out = layer(weight)
        #     out_ab_primes.append(self.forward_transformation(batch_size, grid_size, weight_out))

        # disable grad of implicit layers but keep grad of vanilla layers
        self.set_grad(grad_true = self.net.implicit_layers.parameters(), grad_false = self.net.vanilla_layers.parameters())

        out_ab_primes = []
        if len(self.grid_sizes) > 0:
            assert len(self.grid_sizes) == len(self.net.implicit_layers) == len(self.net.vanilla_layers), f"{len(self.grid_sizes)}, {len(self.net.implicit_layers)}, {len(self.net.vanilla_layers)}"


            assert len(self.grid_sizes) == len(self.net.x_ins), f"{len(self.grid_sizes)}, {len(self.net.x_ins)}"

            for x_in, grid_size, implicit_layer, vanilla_layer in zip(self.net.x_ins, self.grid_sizes, self.net.implicit_layers, self.net.vanilla_layers):
                

                weight, bias = implicit_layer(vanilla_layer.weight, vanilla_layer.bias)

                # weight_out  = implicit_layer(vanilla_layer_weight)
                # weight_out = implicit_layer(vanilla_layer.weight)
                # print(torch.allclose(weight_out, weight_out2))
                out_ab_primes.append(self.forward_transformation(batch_size, x_in, grid_size, weight, bias))
                # out_ab_primes.append(self.forward_transformation(x_in, grid_size, weight, bias))
            


        return out_y, *out_ab_primes
    
    def forward_vanilla(self, batch):

        x, y_true, _ = batch

        y_pred = self.net(x)
        out_y = (y_pred.squeeze(1), y_true.squeeze(1))

        return out_y
    
    def forward_implicit(self, batch):

        x, _, _ = batch
        batch_size = len(x)
            
        out_ab_primes = []
        if len(self.grid_sizes) == 0:
            return out_ab_primes
        
        assert len(self.grid_sizes) == len(self.net.implicit_layers) == len(self.net.vanilla_layers), f"{len(self.grid_sizes)}, {len(self.net.implicit_layers)}, {len(self.net.vanilla_layers)}"
        assert len(self.grid_sizes) == len(self.net.x_ins), f"{len(self.grid_sizes)}, {len(self.net.x_ins)}"

        for x_in, grid_size, implicit_layer, vanilla_layer in zip(self.net.x_ins, self.grid_sizes, self.net.implicit_layers, self.net.vanilla_layers):
            weight, bias = implicit_layer(vanilla_layer.weight, vanilla_layer.bias)
            out_ab_primes.append(self.forward_transformation(batch_size, x_in, grid_size, weight, bias))
        return out_ab_primes

    def log_test_results(self):
        pred_outs = zip(*self.test_step_outs)
        pred_outs = [torch.cat(pred_out).cpu().numpy() for pred_out in pred_outs]

        # tasks = {'classification': self._log_classification, 'regression': self._log_regression}
        # tasks[self.task](*pred_outs)
        self._log_regression(*pred_outs)

    def _log_regression(self, y_preds, y_trues):

        if len(y_preds.shape) == 1:
            y_preds = y_preds.reshape(-1, 1)
            y_trues = y_trues.reshape(-1, 1)

        n_cols = len(y_preds.T)
        fig, axs = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

        if len(y_preds.T) == 1: axs = [axs]

        for ax, y_trues_i, y_preds_i in zip(axs, y_trues.T, y_preds.T):

            l_min, l_max = np.min(y_trues_i), np.max(y_trues_i)*1.1
            ax.plot(y_trues_i, y_preds_i, '.', alpha=0.5)
            ax.plot([l_min, l_max], [l_min, l_max], 'k--')
        fig.supxlabel('True')
        fig.supylabel('Predicted')
        plt.close()
        wandb.log({'regression_results': wandb.Image(fig)})

        print('Logged regression results')
            


        # batch_size = None
        # eps = torch.randn((4,))

        # x_b = x_a = torch.randn_like(x)

        # # Route a: Forward pass, transformation
        # x_a = x
        # out_a = self.net(x_a, batch_size=batch_size)
        # out_a_prime, _ = self.transform(out_a, None, eps)

        # # Route b: Transformation, forward pass
        # x_b = x
        # x_b_prime, _ = self.transform(x_b, None, eps)
        # out_b_prime = self.net(x_b_prime, batch_size=batch_size)

        # assert out_a.shape == x_b.shape
        # assert out_a_prime.shape == out_b_prime.shape
        # assert out_a_prime.shape == x_b_prime.shape


        # y_pred = self.net(x.unsqueeze(1)).squeeze(1).squeeze(1)
        # y_true = y_true.squeeze(1)

        # return (out_a_prime, out_b_prime), (y_pred, y_true)