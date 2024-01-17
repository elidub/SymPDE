import os
import torch.nn as nn
import torch

from model.learner import PredictionLearner, TransformationLearner

from model.networks.mlp import MLP
from model.networks.linear import  LinearP
from data.dataset import FlatDataset
from data.datamodule import BaseDataModule

def setup_model(args):
    net = args.net
    space_length = args.data_kwargs['space_length']

    if net == "TrainP":
        net = LinearP(
            in_features = space_length,
            out_features = space_length,
            bias = False,
            P_init = 'randn',
            train_weights = False,
            train_P = True,
        )
        learner = TransformationLearner

        # TODO: Automatize this, also in the learner
        # transf_kwargs = {'augment' : 'space_translation'}
        transf_kwargs = {'augment' : 'rotation'}
    elif net.startswith("Predict-"):
        if net == "Predict-NoneP":
            net = MLP(
                space_length = space_length, 
                bias = args.bias,
                P_init = 'none',
            )
        elif net == "Predict-CalculatedP":
            net = MLP(
                space_length = space_length, 
                bias = args.bias,
                P_init = 'space_translation',
            )
        elif net == "Predict-TrainedP":
            net = MLP(
                space_length = space_length, 
                bias = args.bias,
                P_init = args.P_pred,
            )
        else:
            raise NotImplementedError(f"Network {net} not implemented")
        learner = PredictionLearner
        transf_kwargs = {'augment' : 'none'}
    else:
        raise NotImplementedError(f"Network {net} not implemented")
    
    dataset = FlatDataset
    
    datamodule = BaseDataModule(
        dataset = dataset,
        data_kwargs = args.data_kwargs,
        transf_kwargs = transf_kwargs,
        data_dir = args.data_dir, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        n_splits = args.n_splits,
        persistent_workers = args.persistent_workers,
    )
    
    criterion = nn.MSELoss()

    if args.train:
        model = learner(net, criterion, lr=args.lr)
        return model, datamodule
    
    # Load model
    assert args.version != None, "Version not specified!"
    ckpt_path = os.path.join(args.log_dir, args.name, args.version, "checkpoints")
    assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found!"
    ckpt = os.listdir(ckpt_path)[0]
    model = learner.load_from_checkpoint(
        os.path.join(ckpt_path, ckpt), net=net, criterion=criterion, lr=args.lr,
        map_location=torch.device('cpu')
    )

    print(f"Loaded model from {ckpt_path}")

    return model, datamodule
