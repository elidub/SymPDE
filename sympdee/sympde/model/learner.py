import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from viz.plot_pde_data import plot_pred
from data.transforms import Transform, TransformRefactored
from data.utils import d_to_coords
from data.pdes import PDEs


class TransformationBlock:
    def __init__(self, pde_name):
        self.rng_a, self.rng_b = torch.Generator(), torch.Generator()
        print('Init rng')

        self.pde = PDEs()[pde_name]

    def augment(self, u, shape, dx = 2., dt = 7.5, epsilons = None, rand = False):
        """
        Augment similar as LPSDA
        """

        batch_size, features = u.shape
        u = u.reshape(batch_size, *shape)

        # Get coordinates
        X = d_to_coords(u[0], dx, dt)
        x, t = X.permute(2, 0, 1)[:2]

        # Augment
        # u, x, t = self.pde.augment(u.clone(), x.clone(), t.clone(), epsilons=epsilons)
        for aug_method, epsilon in zip(self.pde.aug_methods, epsilons):
            if epsilon:
                eps = epsilon * (torch.rand(()) - 0.5) if rand else torch.tensor([epsilon])
                # print(f'Augmenting with {aug_method} with epsilon = {eps}')
                u, x, t = aug_method(u.clone(), x.clone(), t.clone(), eps)

        dx_new = x[0,1] - x[0, 0]
        dt_new = t[1,0] - t[0, 0]
        assert dx_new == dx, f"{dx_new}, {dx}"
        assert dt_new == dt, f"{dt_new}, {dt}"
        u = u.reshape(batch_size, features)


        return u

    def forward_transformation(self, batch_size, x_in, shape, weight, bias):
        self.rng_b.set_state(self.rng_a.get_state())
        assert len(shape) == 2, shape
        shape = {'a':shape[0], 'b':shape[1]}

        epsilons = torch.rand((2,))

        x_a = x_b = torch.randn((batch_size, np.prod(shape['b'])), device = weight.device)
        # x_a = x_b = x_in

        # # Route a: Forward pass, transformation
        out_a = F.linear(x_a, weight, bias)
        out_a_prime = self.augment(out_a, epsilons=epsilons, shape=shape['a'])

        # # Route b: Transformation, forward pass
        x_b_prime = self.augment(x_b, epsilons=epsilons, shape=shape['b'])
        out_b_prime = F.linear(x_b_prime, weight, bias)

        assert out_a_prime.shape == out_b_prime.shape

        # out_a_prime, out_b_prime = torch.zeros_like(x_in), torch.zeros_like(x_in)
        

        return (out_a_prime, out_b_prime)

class Learner(pl.LightningModule, TransformationBlock):
    def __init__(self, net, criterion, pde_name, grid_sizes):
        super().__init__()
        TransformationBlock.__init__(self, pde_name)
        self.net = net
        self.criterion = criterion

        self.x_start = 0
        self.y_start = self.x_end = self.net.time_history
        self.y_end = self.net.time_history+self.net.time_future

        self.grid_sizes = grid_sizes

    def set_grad(self, grad_true, grad_false):
        try:
            for param in grad_true:
                param.requires_grad = True
            for param in grad_false:
                param.requires_grad = False
        except:
            pass


    def forward_vanilla(self, batch, return_pred = True):
        """
        (x, y) refer to (input, target), not to space coordinates
        """

        us, dxs, dts = batch

        # [batch, time, space] -> [batch, space, time]
        us = us.permute(0, 2, 1) 

        #  elect the time history and future for input and target
        x = us[:, :, :self.x_end]        
        y = us[:, :, self.y_start:self.y_end] 

        # Pass the time history through the network
        y_pred = self.net(x, dxs, dts)

        # [batch, space, time] -> [batch, time, space]
        y_pred = y_pred.permute(0, 2, 1)
        y     = y.permute(0, 2, 1)

        # return y_pred, dxs, dts

        out_y = (y_pred, y)
        return out_y

        if return_pred:
            return y_pred, y
        else:
            return y, dxs, dts

    def forward_implicit(self, batch, return_pred = True):
        us, dxs, dts = batch

        # [batch, time, space] -> [batch, space, time]
        us = us.permute(0, 2, 1) 

        #  elect the time history and future for input and target
        x = us[:, :, :self.x_end]   

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
    
    def forward(self, batch):
        out_y = self.forward_vanilla(batch)
        out_ab_primes = self.forward_implicit(batch)
        return out_y, *out_ab_primes

    def step(self, batch, mode="train"):
        # # Forward pass
        # y_pred, y = self.forward(batch)

        # # Loss
        # loss = self.criterion(y_pred, y)

        # # Metrics
        # # Additional metrics can be calculated here

        # # Log
        # self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # return loss, batch, y_pred
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

    def training_step(self, batch, batch_idx):
        loss, batch, (y_pred, y_true) = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch, (y_pred, y_true) = self.step(batch, "val")

        # if batch_idx == 0:
        #     self.log_fig(batch, y_pred, "val")

    def test_step(self, batch, batch_idx):
        loss, batch, (y_pred, y_true) = self.step(batch, "test")
        # self.us_batches.append(batch)
        # self.y_preds_batches.append(y_pred)

        if batch_idx == 0:
            self.log_fig(batch, y_pred, "test")

    def configure_optimizers(self):

        print('Print parameters in configure_optimizers')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print()


        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    # def on_test_epoch_start(self):
    #     # Initialize lists to store inputs and predictions
    #     self.us_batches = []
    #     self.y_preds_batches = []

    # def on_test_epoch_end(self):
    #     return self.us_batches, self.y_preds_batches
    
    def log_fig(self, batch, preds, mode = None, sample_id = 0):
        x_start, x_end, y_start, y_end  = self.x_start, self.x_end, self.y_start, self.y_end

        us, dxs, dts = batch
        dx, dt = dxs[sample_id].cpu().numpy(), dts[sample_id].cpu().numpy()
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

