import os, sys
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision

from emlp.reps import V
from emlp.groups import Z, O
from emlp.datasets import O5Synthetic

from model.learner import CombiLearner
from model.networks.mlp import MLP, CombiMLP, EMLP_wrapper, EMLP_MLP_wrapper
from model.networks.linear import  LinearP
from model.networks.implicit import LinearImplicit
from data.dataset import FlatDataset
from data.datamodule import BaseDataModule
from model.loss import MMDLoss

def load_P_pred(run_id, P_dir = '../logs/store/P/'):
    P = np.load(P_dir + run_id + '.npy')
    P = torch.from_numpy(P).float()
    # P = LinearP.normalize_P(P)
    return P

def load_implicitP_statedict(run_id, P_dir = '../logs/store/implicit_P/', device = 'cpu'):
    statedict = torch.load(P_dir + run_id + '.pt', map_location=torch.device(device))
    return statedict

def find_id_for_P(args):
    df = pd.read_pickle('../logs/store/map_df.pkl')
    args.data_kwargs['grid_size'] = tuple(args.data_kwargs['grid_size'])
    args.transform_kwargs['eps_mult'] = tuple(args.transform_kwargs['eps_mult'])

    for row in df['data_kwargs']:
        if 'grid_size' in row:
            row['grid_size'] = tuple(row['grid_size'])

    for row in df['transform_kwargs']:
        if 'eps_mult' in row:
            row['eps_mult'] = tuple(row['eps_mult'])

    if args.use_P_from_noise:
        data_dir_filter = (df.data_dir == '../data/noise')
        data_kwargs_filter = pd.Series([data_kwarg['grid_size'] == args.data_kwargs['grid_size'] for data_kwarg in df.data_kwargs])
    else:
        data_dir_filter = (df.data_dir == args.data_dir)
        data_kwargs_filter = (df.data_kwargs == args.data_kwargs)

    if hasattr(args, 'svd_rank'):
        if args.svd_rank is not None:
            svd_filter = (df.svd_rank == args.svd_rank)
        else:
            svd_filter = df.svd_rank.isna()
    else:
        svd_filter = df.svd_rank.isna()


    df_selected = df[
        data_kwargs_filter & 
        (df.transform_kwargs == args.transform_kwargs) & 
        (df.seed == args.seed) & 
        data_dir_filter &
        svd_filter
    ]


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
    dataset_name = args.data_dir.split('/')[-1]
    net = args.net
    
    features = np.prod(args.data_kwargs['grid_size'])

    tasks = {
        'ce' : 'classification',
        'mse' : 'regression',
        'mses' : 'regression',
    }
    task = tasks[args.criterion]

    if args.criterion == 'ce':
        out_features = args.n_classes # 10 for MNIST
    else:
        # Manually set out_features for multi-target regression TODO: automate this

        if args.A_low is None:
            if args.y_multi is None:
                out_features = 1
            else:
                out_features = args.y_multi
        else:
            out_features = 2
        assert out_features == args.out_features, f"Expected out_features = {out_features}, got {args.out_features}"

    if args.data_dir == '../data/sine1d':
        transform_type = 'sine1d'
        # args.grid_sizes = [[[1,2,5], [1,2,5]], [[1,1,1], [1,2,5]]]
    elif args.data_dir == '../data/o5synth':
        transform_type = 'o5synth'
        # args.grid_sizes = [[[1,2,5], [1,2,5]], [[1,1,1], [1,2,5]]]
    else:
        raise NotImplementedError(f"transform_type for data_dir={args.data_dir} not implemented")


    if net.startswith("CombiTrain"):
        net = CombiMLP(
            implicit_layer_dims = args.implicit_layer_dims,
            vanilla_layer_dims = args.vanilla_layer_dims,
            bias = args.bias,
            pretrained=args.pretrained,
            forward_type = args.forward_type,
        )
        learner = CombiLearner

    elif net.startswith("EMLP"):

        # Manually select params for EMLP

        if args.data_dir == '../data/sine1d':
            group = Z(7)
            repin, repout =  V(group), V**0
        elif args.data_dir == '../data/o5synth':
            set = O5Synthetic(N = 1)
            group, repin, repout = O(5), set.rep_in, set.rep_out
        else:
            # Test case
            # group, repin, repout = Z(6), V(group), V(group)
            raise NotImplementedError(f"(group, repin, repout) not implemented for data_dir={args.data_dir} not implemented")

        if net == "EMLP":
            net = EMLP_wrapper(
                implicit_layer_dims = args.implicit_layer_dims,
                vanilla_layer_dims = args.vanilla_layer_dims,
                bias = args.bias,
                repin = repin,
                repout = repout,
                group = group,
            )
            learner = CombiLearner
        elif net == "EMLP_MLP":
            net = EMLP_MLP_wrapper(
                implicit_layer_dims = args.implicit_layer_dims,
                vanilla_layer_dims = args.vanilla_layer_dims,
                bias = args.bias,
                repin = repin,
                repout = repout,
                group = group,
            )
            learner = CombiLearner
        else:
            raise NotImplementedError(f"Network {net} not implemented")

    else:
        raise NotImplementedError(f"Network {net} not implemented")
    
    dataset = FlatDataset
    
    datamodule = BaseDataModule(
        dataset = dataset,
        task = task,
        args = args,
        data_kwargs = args.data_kwargs,
        transform_kwargs = args.transform_kwargs,
        data_dir = args.data_dir, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        n_splits = args.n_splits,
        persistent_workers = args.persistent_workers,
    )
    
    criterions = {
        'mse' : nn.MSELoss(),
        'mses' : [(args.lossweight_y, nn.MSELoss())] + [(args.lossweight_o, nn.MSELoss()) for _ in range(len(args.grid_sizes))],
        # 'mses' : [(args.lossweight_y, nn.MSELoss()), (args.lossweight_o, nn.MSELoss())],
        'bce' : nn.BCELoss(),
        'ce'  : nn.CrossEntropyLoss(),
    }

    criterion = criterions[args.criterion.lower()]

    # Load model
    if args.run_id is not None:
        raise NotImplementedError("Loading model from run_id not implemented")

    assert learner == CombiLearner
    model = learner(net, criterion, lr=args.lr, grid_sizes=args.grid_sizes, transform_kwargs=args.transform_kwargs, optimizer_setting=args.optimizer_setting, transform_type=transform_type)

    return model, datamodule