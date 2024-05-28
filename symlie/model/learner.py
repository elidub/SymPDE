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

class BaseLearner(pl.LightningModule):
    def __init__(self, net, criterion, lr, optimizer_setting, **kwargs):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.lr = lr

        self.test_step_outs = []

        self.criterion_alt = True
        self.optimizer_setting = optimizer_setting

        # Change 1: Create a SoftAdapt object (with your desired variant)
        # self.softadapt_object = LossWeightedSoftAdapt(beta=0.1)

        # Change 2: Define how often SoftAdapt calculate weights for the loss components
        # self.epochs_to_make_updates = 5

        # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
        # self.values_of_components = [[], [], []]
        # self.losses = [[], [], []]
        # Initializing adaptive weights to all ones.
        # self.adapt_weights = torch.tensor([1,1,1])


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



        if self.optimizer_setting == 'solo':

            assert len(self.criterion) == 1, self.criterion
            lossweight_y, criterion_y = self.criterion[0]

            out_y = self.forward_vanilla(batch)
            loss_y = criterion_y(*out_y)

            self.log(f"{mode}_loss_y", loss_y, prog_bar=True, on_step=False, on_epoch=True)

            return loss_y, batch, out_y

        loss = 0

        opt_vanilla, opt_implicit = self.optimizers()

        (lossweight_y, criterion_y), (lossweight_o, criterion_o), _ = self.criterion

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
            # if mode == 'train':
                # self.values_of_components[i].append(loss_term.item())
            losses.append(loss_term)
            self.log(f"{mode}_{log_term}", loss_term, prog_bar=True, on_step=False, on_epoch=True)
            loss += lossweight*loss_term
            # loss += self.adapt_weights[i]*loss_term


        # loss = sum([lossweight*loss_term for lossweight, loss_term in zip(self.adapt_weights, losses)])
        

        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        out = out_terms[0] # Only select the prediction of y

        if mode == 'train' and not self.automatic_optimization:

            with torch.autograd.set_detect_anomaly(True):

                (lossweight_y, _), (lossweight_o, _), _ = loss_terms

                # print('Exiting!') ; import sys; sys.exit()

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




            # opt_implicit.zero_grad()

            # loss_vanilla.backward(retain_graph=True)
            # loss_implicit.backward()

            # opt_vanilla.step()
            # opt_implicit.step()



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

    def configure_optimizers_solo(self):

        print('Print parameters in configure_optimizers')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.Adam(self.net.vanilla_layers.parameters(), lr=self.lr)
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

        opt_vanilla = torch.optim.Adam(self.net.vanilla_layers.parameters(), lr=self.lr)
        opt_implicit = torch.optim.Adam(self.net.implicit_layers.parameters(), lr=self.lr)
        return opt_vanilla, opt_implicit
        
        # except:
        #     return opt_vanilla
        
    
    def configure_optimizers(self):

        if self.optimizer_setting == 'solo':
            print('Configuring solo optimizer')
            return self.configure_optimizers_solo()
        elif self.optimizer_setting == 'multi':
            print('Configuring multi optimizer')
            return self.configure_optimizers_multi()


    # def on_train_epoch_start(self) -> None:


    #     if self.current_epoch % self.epochs_to_make_updates == 0 and self.current_epoch != 0:

    #         print(len(self.values_of_components), len(self.values_of_components[0]), len(self.values_of_components[1]), len(self.values_of_components[2]))
    #         print(len(self.losses), len(self.losses[0]), len(self.losses[1]), len(self.losses[2]))

    #         self.adapt_weights = self.softadapt_object.get_component_weights(
    #             torch.tensor(self.values_of_components[0]), 
    #             torch.tensor(self.values_of_components[1]), 
    #             torch.tensor(self.values_of_components[2]),
    #             verbose=True,
    #         )  

    #         # Change 3: Initialize lists to keep track of loss values over the epochs we defined above
    #         self.values_of_components = [[], [], []]
    #         self.losses = [[], [], []]


    # def on_train_epoch_end(self) -> None:

    #     # # TODO: take batch size into account
    #     # self.losses[0].append(torch.mean(torch.tensor(self.values_of_components[0])))
    #     # self.losses[1].append(torch.mean(torch.tensor(self.values_of_components[1])))
    #     # self.losses[2].append(torch.mean(torch.tensor(self.values_of_components[2])))

    #     print("epoch end!")
    #     return super().on_train_epoch_end()
    
    def on_test_end(self):

        if not self.trainer.logger:
            print("No logger, skipping logging")
            return
        
        self.log_test_results()
        
class TransformationLearner(BaseLearner, Transform):
    def __init__(self, net, criterion, lr, grid_size, transform_kwargs):
        BaseLearner.__init__(self, net, criterion, lr)
        Transform.__init__(self, grid_size, **transform_kwargs)

    #     size = torch.prod(torch.tensor(grid_size)).item()
    #     self.generator = self.init_generator_learner(size)

    # def init_generator_learner(self, size):
    #     print(f'Initializing generator with size {size}')
    #     mlp = torchvision.ops.MLP(
    #         in_channels = size + len(self.eps_mult),
    #         hidden_channels = [size, size],
    #     )
    #     return mlp

    def forward(self, batch):

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
        # weight = self.net.weight
        # out_a_tilde = torch.einsum('bi,boi->bo', x_a, weight)
        # out_a_prime_tilde, _ = self.transform(out_a_tilde, centers, eps)

        # out_b_prime_tilde = torch.einsum('bi,boi->bo', x_b_prime, weight)

        assert out_a.shape == x_b.shape
        assert out_a_prime.shape == out_b_prime.shape
        assert out_a_prime.shape == x_b_prime.shape

        criterion_alt = False
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
        run_id = self.trainer.logger.experiment.id
        log_dir = self.trainer.log_dir

        # run_id = 'temp_runid'
        # log_dir = '../logs'

        if hasattr(self.net, 'svd'):
            if self.net.svd: 
                P = self.net.U @ torch.diag(self.net.S) @ self.net.V
                logging_objects = {'P' : P, 'U': self.net.U, 'S': self.net.S, 'V': self.net.V}
            else:
                logging_objects = {'P': self.net.P}
            save_format = 'numpy'
        elif hasattr(self.net, 'implicit_P'):
            logging_objects = {'implicit_P': self.net.implicit_P.state_dict()}
            save_format = 'state_dict'
        else:
            raise NotImplementedError(f"Logging not implemented for {self.net}")

        for key, value in logging_objects.items():
            print(f'Logging {key}')
            store_dir = os.path.join(log_dir, 'store', key)
            os.makedirs(store_dir, exist_ok=True)
            if save_format == 'numpy':
                np.save(os.path.join(store_dir, f'{run_id}.npy'), value.cpu().numpy())
            elif save_format == 'state_dict':
                torch.save(value, os.path.join(store_dir, f'{run_id}.pt'))

class MLPLearner(BaseLearner):
    def __init__(self, net, criterion, lr):
        super().__init__(net, criterion, lr, optimizer_setting='solo')

        self.criterion_alt = False

    def forward(self, batch):

        x, y_true, _ = batch

        y_pred = self.net(x)
        out_y = (y_pred.squeeze(1), y_true.squeeze(1))

        return out_y
    
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

# class TransformationBlock(TransformRefactored):
#     def __init__(self, transform_kwargs, **kwargs):
#         TransformRefactored.__init__(self, eps_mult = transform_kwargs['eps_mult'])

#     def forward_transformation(self, batch_size, shape, weight, bias):

#         assert len(shape) == 2, shape
#         shape = {'a':shape[0], 'b':shape[1]}

#         eps = torch.randn((4,))
        
#         x_a = x_b = torch.randn((batch_size, np.prod(shape['b'])), device = weight.device)

#         # Route a: Forward pass, transformation
#         out_a = F.linear(x_a, weight, bias)
#         out_a_prime = self.transform(out_a, eps, shape=shape['a'])

#         # Route b: Transformation, forward pass
#         x_b_prime = self.transform(x_b, eps, shape=shape['b'])
#         out_b_prime = F.linear(x_b_prime, weight, bias)

#         assert out_a_prime.shape == out_b_prime.shape

#         return (out_a_prime, out_b_prime)

class TransformationBlock(TransformRefactored):
    def __init__(self, transform_kwargs, **kwargs):
        TransformRefactored.__init__(self, eps_mult = transform_kwargs['eps_mult'])

        self.rng_a, self.rng_b = torch.Generator(), torch.Generator()

        print('Init rng')

    def forward_transformation(self, batch_size, x_in, shape, weight, bias):

        # print('batch_size', batch_size)
        # print('x_in', x_in.shape)
        # print('shape', shape)
        # print('weight', weight.shape)
        # print('bias', bias.shape)

        # seed = torch.randint(0, 100000, (1,)).item()
        # self.rng_a.manual_seed(seed)
        # self.rng_b.manual_seed(seed)

        self.rng_b.set_state(self.rng_a.get_state())

        # if weight.shape == (1,10):
        #     weight = torch.ones(1,10)
        # elif weight.shape == (10,10):
        #     weight = torch.eye(10)
        # else:
        #     raise NotImplementedError(f"Weight shape {weight.shape} not implemented")

        assert len(shape) == 2, shape
        shape = {'a':shape[0], 'b':shape[1]}

        x_a = x_b = torch.randn((batch_size, np.prod(shape['b'])), device = weight.device)
        # x_a = x_b = x_in
        # x_a = x_b = batch_size.clone()

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
        # print('Combilearner init')
        BaseLearner.__init__(self, net, criterion, lr, optimizer_setting)
        TransformationBlock.__init__(self, transform_kwargs)
        # super(BaseLearner, self).__init__(net, criterion, lr)
        # super(TransformationBlock, self).__init__(transform_kwargs)
        # super().__init__(net, criterion, lr, transform_kwargs)
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