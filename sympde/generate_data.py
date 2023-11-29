
import torch
import os
import pytorch_lightning as pl
import argparse
import logging

from misc.utils import save_obj
from data.solve import SolvePDE
from data.pdes import PDEs


def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='Generating PDE data')

    parser.add_argument("--data_dir", type=str, default="../data/dev", help="Path to data directory")

    parser.add_argument("--Lmax", type = int, default = 64, help = "Length of spatial domain")
    parser.add_argument("--Tmax", type = int, default = 40, help = "Length of temporal domain")
    parser.add_argument("--Nx", type = int, default = 256, help = "Number of spatial grid points")
    parser.add_argument("--Nt", type = int, default = 40, help = "Number of temporal grid points")
    parser.add_argument("--tol", type = float, default = 1e-6, help = "Tolerance for ODE solver")

    parser.add_argument("--pde_names", default = ["Pde1","KdV"], nargs='+', help = "List of name of the PDEs to generate data for")
    parser.add_argument("--n_splits", default=[160,20,20], nargs='+', help="Train, val, test split")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):

    pdes = PDEs()

    n_train, n_val, n_test = [int(n_split) for n_split in args.n_splits]
    splits = {'train':{'n_samples':n_train}, 'val':{'n_samples':n_val}, 'test':{'n_samples':n_test}}

    pde_data = SolvePDE(Lmax=args.Lmax, Tmax=args.Tmax, Nx=args.Nx, Nt=args.Nt, tol=args.tol)

    for pde_name in args.pde_names:
        for split, split_dict in splits.items():
            os.makedirs(os.path.join(args.data_dir, split), exist_ok=True)
            us, dxs, dts = pde_data.generate_data(pdes[pde_name], N_samples = split_dict['n_samples'], tqdm_desc=f"Generating {split} data for {pde_name}!")
            save_obj((us, dxs, dts), os.path.join(args.data_dir, split, pde_name))


if __name__ == '__main__':
    args = parse_options()
    print(args)
    main(args)