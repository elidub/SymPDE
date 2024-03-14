import os, sys
import torch.nn as nn
import torch
import numpy as np
import pandas as pd

from model.learner import PredictionLearner, TransformationLearner
from model.networks.mlp import MLP
from model.networks.linear import  LinearP
from model.networks.implicit import LinearImplicit
from data.dataset import FlatDataset
from data.datamodule import BaseDataModule

def load_P_pred(run_id, P_dir = '../logs/store/P/'):
    P = np.load(P_dir + run_id + '.npy')
    P = torch.from_numpy(P).float()
    # P = LinearP.normalize_P(P)
    return P

def load_implicitP_statedict(run_id, P_dir = '../logs/store/implicit_P/'):
    statedict = torch.load(P_dir + run_id + '.pt')
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

    if net.startswith("Train"):
        if net == "TrainP":
            net = LinearP(
                in_features = features,
                out_features = features,
                bias = False,
                device = args.device,
                P_init = 'randn',
                train_weights = False,
                train_P = True,
                svd_rank = args.svd_rank,
            )
        elif net == "TrainImplicitP":
            net = LinearImplicit(
                in_features = features,
                out_features = features,
                bias = False,
                hidden_implicit_layers = args.hidden_implicit_layers,
                device = args.device,
                train_weights = False,
                train_P = True,
            )
        learner = TransformationLearner

    elif net.startswith("Predict-"):
        if net == "Predict-NoneP":
            net = MLP(
                in_features = features, 
                out_features = out_features,
                bias = args.bias,
                n_hidden_layers = args.n_hidden_layers,
                device = args.device,
                P_init = 'none',
            )
        elif net == "Predict-CalculatedP":
            net = MLP(
                in_features = features, 
                out_features = out_features,
                bias = args.bias,
                n_hidden_layers = args.n_hidden_layers,
                device = args.device,
                P_init = 'space_translation',
            )
        elif net == "Predict-TrainedP":
            assert args.use_P_from_noise == False
            P_pred = load_P_pred(find_id_for_P(args)).to(args.device)
            net = MLP(
                in_features = features, 
                out_features = out_features,
                bias = args.bias,
                n_hidden_layers = args.n_hidden_layers,
                device = args.device,
                P_init = P_pred,
            )
        elif net == "Predict-NoiseTrainedP":
            args.use_P_from_noise = True
            assert args.use_P_from_noise == True

            # P_pred = load_P_pred(find_id_for_P(args)).to(args.device)

            # P_pred = load_P_pred('7u75g6ai').to(args.device) # debug normalize_P
            # P_pred = load_P_pred('3uvrx8mf').to(args.device) # debug WITHOUT normalize_P
            manual_id = 'a8usb5wi' # debug normalize_P again
            manual_id = 'y2xmhybx' # debug WITHOUT normalize_P again

            P_pred = load_P_pred(manual_id).to(args.device) 
            net = MLP(
                in_features = features, 
                out_features = out_features,
                bias = args.bias,
                n_hidden_layers = args.n_hidden_layers,
                device = args.device,
                P_init = P_pred,
                linearmodules=[LinearP, nn.Linear],
            )

        elif net == "Predict-NoiseTrainedImplicitP":
            args.use_P_from_noise = True
            assert args.use_P_from_noise == True
            # statedict_implicitP = load_implicitP_statedict(find_id_for_P(args))
            statedict_implicitP = load_implicitP_statedict('be1v5f84')
            print('statedict_implicitP', statedict_implicitP.keys())
            net = MLP(
                in_features = features, 
                out_features = out_features,
                bias = args.bias,
                n_hidden_layers = args.n_hidden_layers,
                device = args.device,
                P_init = statedict_implicitP,
                linearmodules = [LinearImplicit, nn.Linear],
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
        # 'mses' : [(args.lossweight_o, nn.MSELoss()), (args.lossweight_dg, nn.MSELoss()), (0., nn.MSELoss()), (0., nn.MSELoss())],
        'mses' : [(args.lossweight_o, nn.MSELoss()), (args.lossweight_dg, nn.MSELoss())],
        'bce' : nn.BCELoss(),
        'ce'  : nn.CrossEntropyLoss(),
    }

    criterion = criterions[args.criterion.lower()]

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
                os.path.join(ckpt_path, ckpt), net=net, criterion=criterion, lr=args.lr, task=task,
                map_location=torch.device('cpu')
            )
        print(f"Loaded model from {ckpt_path}")

    else:
        if learner == TransformationLearner:
            model = learner(net, criterion, lr=args.lr, grid_size=args.data_kwargs['grid_size'], transform_kwargs=args.transform_kwargs)
        elif learner == PredictionLearner:
            model = learner(net, criterion, lr=args.lr, task=task)
        else:
            raise NotImplementedError(f"Network {net} not implemented")

    return model, datamodule