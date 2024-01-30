import os, sys
import torch.nn as nn
import torch
import numpy as np
import pandas as pd

from model.learner import PredictionLearner, TransformationLearner
from model.networks.mlp import MLP
from model.networks.linear import  LinearP
from data.dataset import FlatDataset
from data.datamodule import BaseDataModule

def load_P_pred(run_id, P_dir = '../logs/store/P/'):
    P = np.load(P_dir + run_id + '.npy')
    P = torch.from_numpy(P).float()
    P = LinearP.normalize_P(P)
    return P
# def find_id_for_P(args):
#     df = pd.read_pickle('../logs/store/results_df.pkl')
#     args.data_kwargs['grid_size'] = tuple(args.data_kwargs['grid_size'])

#     for row in df['data_kwargs']:
#         if 'grid_size' in row:
#             row['grid_size'] = tuple(row['grid_size'])

#     df_selected = df[(df.data_kwargs == args.data_kwargs) & (df.transform_kwargs == args.transform_kwargs) & (df.seed == args.seed)]

#     if len(df_selected) == 0:
#         raise ValueError('No results found for the given arguments')
#     elif len(df_selected) > 1:
#         print(df_selected['run_id'])
#         raise ValueError('Multiple results found for the given arguments')
#     else:
#         run_id = df_selected.iloc[0]['run_id']
#         print(f"Found run_id {run_id}")
#         return run_id

def find_id_for_P(args):
    df = pd.read_pickle('../logs/store/map_df.pkl')
    args.data_kwargs['grid_size'] = tuple(args.data_kwargs['grid_size'])

    for row in df['data_kwargs']:
        if 'grid_size' in row:
            row['grid_size'] = tuple(row['grid_size'])

    df_selected = df[(df.data_kwargs == args.data_kwargs) & (df.transform_kwargs == args.transform_kwargs) & (df.seed == args.seed)]

    if len(df_selected) == 0:
        raise ValueError('No results found for the given arguments')
    elif len(df_selected) > 1:
        print(df_selected['run_id'])
        raise ValueError('Multiple results found for the given arguments')
    else:
        run_id = df_selected.iloc[0]['run_id']
        print(f"Found run_id {run_id}")
        return run_id

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
            P_pred = load_P_pred(find_id_for_P(args)).to(args.device)
            net = MLP(
                features = features, 
                bias = args.bias,
                device = args.device,
                P_init = P_pred,
            )
        elif net == "Predict-TrainedP-check":
            find_id_for_P(args)
            sys.exit()
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
        print(ckpt_path)
        assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found!"
        ckpt = os.listdir(ckpt_path)[0]
        print(ckpt)
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