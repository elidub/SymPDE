import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import cycle, islice

from data.generate_2d import Create2dData
from misc.utils import Args
from model.setup import find_id_for_P, load_P_pred


def plot_data(dataset, data_kwargs_list=['data_kwargs_show', 'data_kwargs'], N_plot=5, l=1):
    for data_kwargs in data_kwargs_list:
        create_data = Create2dData(dataset['create_sample_func'], dataset[data_kwargs], dataset['transform_kwargs'])
        out = create_data(N = N_plot)
        x, y = out['x'].reshape(N_plot, *dataset[data_kwargs]['grid_size']), out['y']
        dataset['plot_func'](x, y, l)

def assert_unique(df_map_new):
    df_map_new = df_map_new.copy()
    del df_map_new['run_id']
    for col in df_map_new.columns:
        df_map_new[col] = df_map_new[col].astype(str)
    assert len(df_map_new) == len(df_map_new.drop_duplicates())

def stringify_dict(d, stringify):
    # ['seed', 'batch_size', 'noise_std', 'lr', 'test_loss']
    df_map_new = d.copy()
    for col in df_map_new.columns:
        if col not in stringify: continue
        df_map_new[col] = df_map_new[col].astype(str)
    return df_map_new


def pivot(d, columns = ['batch_size', 'lr', 'net'], index = 'seed', values = 'test_loss'):
    if type(columns) == str: columns = [columns]
    if type(values) == str:  values = [values]
    if type(index) == str:   index = [index]

    d = d[columns + index + values].sort_values(by= columns + index).reset_index(drop=True).pivot(index=index, columns=columns, values=values)
    return d

def plot_pivot(d=None, columns=None,d_pivot=None, figsize=(8, 4), logx = False, legend_loc = None, suptitle = None):

    if d_pivot is None:
        assert (d is not None) and (columns is not None)
    else:
        assert (d is None) and (columns is None)

    d_pivot = pivot(d=d, columns=columns) if d_pivot is None else d_pivot

    net_names = d_pivot.columns.get_level_values(3).unique()
    net_names_order = ['Vanilla', 'Pre-calculated', 'Trained', 'Trained with noise']
    net_names = [net for net in net_names_order if net in net_names]

    unstack = lambda d, metric: d.apply(metric).unstack().reset_index(level=0, drop = True)[net_names]
    d_mean, d_std = unstack(d_pivot, pd.Series.mean), unstack(d_pivot, pd.Series.std)


    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(d_mean)))


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
    d_mean.plot(kind='bar', yerr=d_std, ax = ax)
    if logx: ax.set_yscale('log')
    ax.legend(loc=legend_loc)

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, tight_layout=True)
    # for ax in axs:
    #     d_mean.plot(kind='bar', yerr=d_std, ax = ax, legend = False)
    # axs[0].legend(loc=legend_loc)
    # axs[1].set_yscale('log')

    fig.suptitle(suptitle)

    plt.show()


def rename_net(d_pivot, level = 3, index = 0):
    renames = {
        'Predict-NoneP': 'Vanilla', 
        'Predict-CalculatedP': 'Pre-calculated', 
        'Predict-TrainedP': 'Trained',
        'Predict-NoiseTrainedP': 'Trained with noise'
    }
    
    d_pivot = d_pivot.rename(columns=renames)
    
    # new_cols = d_pivot.columns.reindex(['Vanilla', 'Trained', 'Pre-calculated'], level = level)
    new_cols = d_pivot.columns.reindex(list(renames.values()), level = level)
    d_pivot = d_pivot.reindex(columns=new_cols[index])
    return d_pivot

def plot_seeds_and_Ps(d_pivot, P_plots, disable_ticks=False):
    mosaic_bottom_list = list(map(chr, range(ord('B'), ord('B')+len(P_plots))))
    mosaic_top, mosaic_bottom = 'A'*len(P_plots), ''.join(mosaic_bottom_list)
    fig, axs = plt.subplot_mosaic(f"{mosaic_top};{mosaic_bottom}", figsize = (3*len(P_plots),6), tight_layout = True, gridspec_kw = dict(height_ratios = [0.5, 1]))
    net_colors = {'Vanilla': 'C0', 'Pre-calculated': 'C1', 'Trained': 'C2'}

    ax = axs['A']

    for i, (seed, row) in enumerate(d_pivot.iterrows()):
        vals=row.values
        ax.plot(vals, label = seed, marker = None, ls = '--', alpha = 0.1, color = 'k')

    for i, (labels, test_losses) in enumerate(d_pivot.T.iterrows()):
        vals = test_losses.values
        _, net = labels
        ax.scatter(np.full_like(vals, i), vals, alpha = 0.5, marker = 'x', color = net_colors[net])

    nets = d_pivot.columns.get_level_values(1)
    ax.set_xticks(np.arange(len(nets)), nets)
    ax.set_ylabel('Test loss')

    for (net, P), ax_i in zip(P_plots.items(), mosaic_bottom_list):
        ax = axs[ax_i]
        ax.imshow(P)
        ax.patch.set_edgecolor(net_colors[net])

        ax.patch.set_linewidth(5)  

        # Disable ticks
        if not disable_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()

def add_df_map_new(df_map_new, filename = '../logs/store/map_df.pkl'):
    df_map_old = pd.read_pickle(filename)
    df_map = pd.concat([df_map_old, df_map_new]).drop_duplicates(subset=['run_id']).reset_index(drop=True)
    assert_unique(df_map)
    df_map.to_pickle(filename)
    return df_map

def assert_columns_same(d, columns, dataset=None):
    vals_same = {}
    for col in columns:
        col_vals = d[col].values
        for val in col_vals:
            assert str(val) == str(col_vals[0]), f"Expected {col} = {col_vals[0]}, got {col} = {val}"
        col_val = col_vals[0]
        
        # if dataset is not None:
        #     assert col_val == dataset[col], f"Expected {col} = {dataset[col]}, got {col} = {col_val}"
        
        
        vals_same[col] = col_val


    return vals_same

def get_and_check_Ps(seeds, map_kwargs, use_P_from_noise=False):
    Ps = []
    for seed in seeds:

        args = Args(**dict(seed = seed, use_P_from_noise=use_P_from_noise, **map_kwargs))

        run_id = find_id_for_P(args)
        P = load_P_pred(run_id)
        Ps.append(P)
    Ps = torch.stack(Ps)
    return Ps