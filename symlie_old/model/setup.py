import os
import torch.nn as nn
import torch

from model.learner import Learner, FlatLearner, FlatYLearner

from model.networks.mlp import MLP, MLPTorch
from model.networks.linear import MyLinear, MyLinearPw
from model.networks.convs import MyConv1d
from data.dataset import MnistDataModule, FlatDataModule, FlatYDataModule

def setup_model(args):
    net = args.net
    space_length = args.data_kwargs['space_length']

    linearmodules = []
    for linearmodule in args.linearmodules:
        if linearmodule == "MyLinear":
            linearmodules.append(MyLinear)
        elif linearmodule == "MyLinearPw":
            linearmodules.append(MyLinearPw)
        elif linearmodule == "nn.Linear":
            linearmodules.append(nn.Linear)
        else:
            raise NotImplementedError(f"Linearmodule {linearmodule} not implemented") 

    if net == "MLP":
        net = MLP(transform_type=args.transform_type,space_length=space_length, linearmodules=linearmodules, bias=args.bias)
        datamodule = NotImplemented
        learner = Learner
    elif net == "MLPTorch":
        net = MLPTorch(space_length=space_length)
        datamodule = NotImplemented
        learner = Learner
    elif net == "Conv1d":
        net = MyConv1d(
            in_channels = 1,
            out_channels = 1,
            kernel_size = 3,
            padding = 1,
            bias = False,
            padding_mode = 'circular',
        )
        learner = FlatLearner
        datamodule = NotImplemented
    elif net == "LearnPw":
        net = MyLinearPw(
            in_features = space_length,
            out_features = space_length,
            bias = False,
            # transform_type = args.transform_type,
            transform_type = 'train_Pw',
        )
        learner = FlatLearner
        datamodule = FlatYDataModule
    elif net == "LearnY":
        net = MLP(
            transform_type='none',
            train_layer = True,
            space_length=space_length, 
            linearmodules=[MyLinearPw, nn.Linear], 
            bias=args.bias
        )
        learner = FlatYLearner
        datamodule = FlatYDataModule
    elif net == "LearnedY":
        net = MLP(
            transform_type='trained',
            train_layer = True,
            w_index_trained = args.w_index_trained,
            space_length=space_length, 
            linearmodules=[MyLinearPw, nn.Linear], 
            bias=args.bias
        )
        learner = FlatYLearner
        datamodule = FlatYDataModule
    elif net == "CalculatedY":
        net = MLP(
            transform_type='space_translation',
            train_layer = True,
            space_length=space_length, 
            linearmodules=[MyLinearPw, nn.Linear], 
            bias=args.bias
        )
        learner = FlatYLearner
        datamodule = FlatYDataModule
    else:
        raise NotImplementedError(f"Network {net} not implemented")
    
    datamodule = datamodule(
        augment = args.transform_type,
        data_kwargs = args.data_kwargs,
        data_dir = args.data_dir, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        n_splits = args.n_splits,
        persistent_workers = args.persistent_workers,
    )

    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    if args.train:
        model = learner(net, criterion, lr=args.lr)
        return model, datamodule
    
    # Load model
    ckpt_path = os.path.join(args.log_dir, args.name, args.version, "checkpoints")
    assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found!"
    assert args.version != None, "Version not specified!"
    ckpt = os.listdir(ckpt_path)[0]
    model = learner.load_from_checkpoint(
        os.path.join(ckpt_path, ckpt), net=net, criterion=criterion, lr=args.lr,
        map_location=torch.device('cpu')
    )

    return model, datamodule
