from data.generate_data import GeneratePDEData
from data.pde_collection import CollectionPDE_Pseudospectral
from viz.plot_pde_data import plot_1d, plot_1ds, plot_1d_dict
from data.utils import save_obj, load_obj

import torch
import os
import pytorch_lightning as pl
import argparse
import logging


from data.dataset import PDEDataset, PDEDataModule
from model.setup import setup_model
from data.lpda_data_aug import SpaceTranslate, Scale, Galileo

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='Generating PDE data')

    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")

    parser.add_argument("--Lmax", type = int, default = 64, help = "Length of spatial domain")
    parser.add_argument("--Tmax", type = int, default = 40, help = "Length of temporal domain")
    parser.add_argument("--Nx", type = int, default = 256, help = "Number of spatial grid points")
    parser.add_argument("--Nt", type = int, default = 40, help = "Number of temporal grid points")
    parser.add_argument("--tol", type = float, default = 1e-6, help = "Tolerance for ODE solver")

    parser.add_argument("--pde_names", default = ["pde1","KdV"], nargs='+', help = "List of name of the PDEs to generate data for")
    parser.add_argument("--n_splits", default=[160,20,20], nargs='+', help="Train, val, test split")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):

    pde_collection = CollectionPDE_Pseudospectral(L = args.Lmax).collection
    pde_data = GeneratePDEData(Lmax=args.Lmax, Tmax=args.Tmax, Nx=args.Nx, Nt=args.Nt, tol=args.tol)

    n_train, n_val, n_test = [int(n_split) for n_split in args.n_splits]
    splits = {'train':{'n_samples':n_train}, 'val':{'n_samples':n_val}, 'test':{'n_samples':n_test}}

    for pde_name, pde_func in pde_collection.items():

        if pde_name not in args.pde_names:
            continue
        for split, split_dict in splits.items():
            os.makedirs(os.path.join(args.data_dir, split), exist_ok=True)
            us, dxs, dts = pde_data.generate_data(pde_func, N_samples = split_dict['n_samples'], tqdm_desc=f"Generating {split} data for {pde_name}!")
            save_obj((us, dxs, dts), os.path.join(args.data_dir, split, pde_name))


if __name__ == '__main__':
    args = parse_options()
    print(args)
    main(args)