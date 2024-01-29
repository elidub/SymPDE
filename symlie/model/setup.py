import os
import torch.nn as nn
import torch
import numpy as np

from model.learner import PredictionLearner, TransformationLearner

from model.networks.mlp import MLP
from model.networks.linear import  LinearP
from data.dataset import FlatDataset
from data.datamodule import BaseDataModule

def setup_model(args):
    net = args.net
    
    features = np.prod(args.data_kwargs['grid_size'])

    if net == "TrainP":
        net = LinearP(
            in_features = features,
            out_features = features,
            bias = False,
            device = args.device,
            P_init = 'randn',
            train_weights = False,
            train_P = True,
        )
        learner = TransformationLearner

    elif net.startswith("Predict-"):
        if net == "Predict-NoneP":
            net = MLP(
                features = features, 
                bias = args.bias,
                device = args.device,
                P_init = 'none',
            )
        elif net == "Predict-CalculatedP":
            net = MLP(
                features = features, 
                bias = args.bias,
                device = args.device,
                P_init = 'space_translation',
            )
        elif net == "Predict-TrainedP":
            net = MLP(
                features = features, 
                bias = args.bias,
                device = args.device,
                P_init = args.P_pred,
            )
        else:
            raise NotImplementedError(f"Network {net} not implemented")
        learner = PredictionLearner
    else:
        raise NotImplementedError(f"Network {net} not implemented")
    
    dataset = FlatDataset
    
    datamodule = BaseDataModule(
        dataset = dataset,
        data_kwargs = args.data_kwargs,
        transform_kwargs = args.transform_kwargs,
        data_dir = args.data_dir, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        n_splits = args.n_splits,
        persistent_workers = args.persistent_workers,
    )
    
    criterion = nn.MSELoss()

    # Load model
    if args.run_id is not None:
        assert args.version != None, "Version not specified!"
        # ckpt_path = os.path.join(args.log_dir, args.name, args.version, "checkpoints")
        ckpt_path = os.path.join(args.log_dir, 'symlie', args.run_id, "checkpoints")
        assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found!"
        ckpt = os.listdir(ckpt_path)[0]
        if learner == TransformationLearner:
            model = learner.load_from_checkpoint(
                os.path.join(ckpt_path, ckpt), net=net, criterion=criterion, lr=args.lr, grid_size=args.data_kwargs['grid_size'], transform_kwargs=args.transform_kwargs,
                map_location=torch.device('cpu')
            )
        if learner == PredictionLearner:
            model = learner.load_from_checkpoint(
                os.path.join(ckpt_path, ckpt), net=net, criterion=criterion, lr=args.lr,
                map_location=torch.device('cpu')
            )
        print(f"Loaded model from {ckpt_path}")

    else:
        if learner == TransformationLearner:
            model = learner(net, criterion, lr=args.lr, grid_size=args.data_kwargs['grid_size'], transform_kwargs=args.transform_kwargs)
        elif learner == PredictionLearner:
            model = learner(net, criterion, lr=args.lr)
        else:
            raise NotImplementedError(f"Network {net} not implemented")

    return model, datamodule