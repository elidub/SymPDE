import os
import torch.nn as nn
import torch

from model.networks.fno import FNO1d
from model.learner import Learner

def setup_model(args):
    net = args.net

    if net == "FNO1d":

        time_history = 10
        time_future  = 5

        net = FNO1d(
            time_history = time_history,
            time_future  = time_future,
        )
        criterion = nn.MSELoss(reduction = 'none')
    else:
        raise NotImplementedError
    
    model = Learner(net, criterion)

    return model

# def FNO_forward(model, batch):
#     x, dx, dt = batch
#     y_hat = model.net(x)
#     return y_hat, x