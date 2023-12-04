from time import time

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from numpy import cos, sin, arange, trace, pi, newaxis, sum, eye, array, mgrid, diag
from numpy.linalg import norm, matrix_power
from numpy.random import randn, rand, seed

from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, legend, show, xlim, ylim, subplot, grid, savefig, imshow, colorbar, tight_layout, text, xticks, yticks


class Trainer():
    def __init__(self, model, device, optimizer, dataset_class = None, 
                 train_loader=None, test_loader=None,
                 batch_size = 64, test_batch_size = 1000,loss_func = F.nll_loss,
                ):
        """
        usage:
            t = trainer(...)
            t.fit(epochs)
            
        methods:
            .fit(epochs) : train + test; print results; stores results in trainer.history <dict>
            .train(epoch)
            .test()
        """
        self.device = device #torch.device(device)
        self.optimizer = optimizer
        self.model = model
        self.loss_func = loss_func

        self.scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        self.history = {'train loss':[], 'test loss':[], 'train acc':[], 'test acc':[], 'train time':[]}
        
        if dataset_class:
            self.make_dataloaders(dataset_class, batch_size, test_batch_size)
        else:
            self.train_loader = train_loader 
            self.test_loader = test_loader
        
    def progbar(self,percent, N=10):
        n = int(percent//N)
        return '[' + '='*n + '>' +'.'*(N-n-1) +']'
    
    def train(self,epoch):
        self.model.train()
        training_loss = 0
        correct = 0
        t0 = time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if type(data)==list:
                data = [d.to(self.device) for d in data]
            else:
                data = data.to(self.device)
            if type(target)==list:
                target = [d.to(self.device) for d in target]
            else:
                target = target.to(self.device)
                
#             data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            training_loss += loss.sum().item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward(retain_graph=True)
            self.optimizer.step()
            if batch_idx % 10 == 0:
                perc = 100. * batch_idx / len(self.train_loader)
                t1 = time()
                # print('Train Epoch: {} {} {:.1f}s [{}/{} ({:.0f}%)]\tLoss: {:.4g}'.format(
                    # epoch, self.progbar(perc), t1-t0,
                    # batch_idx * len(data), len(self.train_loader.dataset), # n/N
                    # perc, # % passed
                    # loss.item()), end='\r')
        
        training_loss /= len(self.train_loader.dataset)
        acc = correct / len(self.train_loader.dataset)    
        # print('Training: loss: {:.4g}, Acc: {:.2f}%'.format(training_loss, 100.*acc))
            
        return {'loss':training_loss, 'acc':acc , 'time':t1-t0}
                
        
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if type(data)==list:
                    data = [d.to(self.device) for d in data]
                else:
                    data = data.to(self.device)
                if type(target)==list:
                    target = [d.to(self.device) for d in target]
                else:
                    target = target.to(self.device)
                #data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_func(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        
        test_loss /= len(self.test_loader.dataset)
        test_acc = correct / len(self.test_loader.dataset)
        
        # print('Test loss: {:.4g}, Test acc.: {:.2f}%'.format( test_loss, 100.*test_acc))
        return {'loss':test_loss, 'acc':test_acc}
    
    def fit(self,epochs=1):
        for epoch in tqdm(range(1, epochs + 1)):
            r = self.train(epoch)
            self.history['train loss'] += [r['loss']]
            self.history['train acc'] += [r['acc']]
            self.history['train time'] += [r['time']]
            
            r = self.test()
            self.history['test loss'] += [r['loss']]
            self.history['test acc'] += [r['acc']]
            self.scheduler.step()