import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List

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

def plot_pivot(d_pivot, step: int, figsize=(8, 4), logx = False, legend_loc = None, suptitle = None):

    if step == 2:
        net_names = d_pivot.columns.get_level_values(3).unique()
    else:
        net_names = slice(None)

    unstack = lambda d, metric: d.apply(metric).unstack().reset_index(level=0, drop = True)[net_names]
    d_mean, d_std = unstack(d_pivot, pd.Series.mean), unstack(d_pivot, pd.Series.std)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
    d_mean.plot(kind='bar', yerr=d_std, ax = ax, legend = True)#.legend(loc=legend_loc)
    if logx: ax.set_yscale('log')

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, tight_layout=True)
    # for ax in axs:
    #     d_mean.plot(kind='bar', yerr=d_std, ax = ax, legend = False)
    # axs[0].legend(loc=legend_loc)
    # axs[1].set_yscale('log')

    fig.suptitle(suptitle)

    plt.show()

def aggregate_dataset(datasets: List[str], df: pd.DataFrame, step: int, group_params: List[str], hyper_params: List[str], check_df: str, default_log: List[str] = [], plot_single_pivot: bool = False):
    df_map_columns = ['run_id', 'tags', 'data_kwargs', 'transform_kwargs', 'seed', 'data_dir', 'svd_rank']

    d_pivots = {}
    index_columns = ['dataset_name'] + group_params
    dds, dds_mean, dds_std, mins = [], [], [], []

    for dataset_name in datasets:



        # Filter step on df
        if step == 1:
            d = df[df['tags'].astype(str).str.contains(f"'{dataset_name}'")].reset_index(drop=True)
            d = d[~d['tags'].astype(str).str.contains(f'predict')].reset_index(drop=True)
        elif step == 2:
            d = df[df['tags'].astype(str).str.contains(f'{dataset_name}-predict')].reset_index(drop=True)

        d = stringify_dict(d, group_params) # Striningify group_params for unique indexing
        ds = {group : d_group for group, d_group in d.groupby(group_params)} if group_params else {0 : d}


        for group, d in ds.items():

            d = d.reset_index(drop=True)

            for row in d['data_kwargs']:
                if 'grid_size' in row:
                    row['grid_size'] = tuple(row['grid_size'])
            for row in d['transform_kwargs']:
                if 'eps_mult' in row:
                    row['eps_mult'] = [float(x) for x in row['eps_mult']]

            d = d.drop_duplicates(subset=hyper_params + ['seed', 'test_loss']) #TODO: why is this necessary?
            map_kwargs = assert_columns_same(d, ['data_kwargs', 'transform_kwargs', 'data_dir'])


            d_pivot = pivot(d, columns=hyper_params)
            if step == 2:
                d_pivot = rename_net(d_pivot)
            d_pivots[(dataset_name, group)] = dict(d_pivot=d_pivot, map_kwargs=map_kwargs)

            dd = pd.DataFrame(data = [f'{mean:.2e} ± {std:.2e}' for mean, std in zip(d_pivot.mean().values, d_pivot.std().values)], index = d_pivot.columns).T
            dd_mean = pd.DataFrame(data = [mean for mean in d_pivot.mean().values], index = d_pivot.columns).T
            dd_std = pd.DataFrame(data = [std for std in d_pivot.std().values], index = d_pivot.columns).T

            for dd_i, dds_i in zip([dd, dd_mean, dd_std], [dds, dds_mean, dds_std]):

                dd_i[index_columns] = [dataset_name] + list(group)
                dd_i = dd_i.set_index(index_columns)
                dds_i.append(dd_i)

            if step == 1 and check_df != 'inspect':
                hparams_min = d_pivot.mean().idxmin()
                assert len(hparams_min) == 3
                df_map_new = d[d[hyper_params[0]] == hparams_min[1]][df_map_columns]
                add_df_map_new(df_map_new)
                print(f'{dataset_name=}, {group=}, {hparams_min=}')
                mins.append((dataset_name, group, hparams_min))

            # continue
            logx = True if dataset_name in default_log else False
            suptitle = dataset_name + '\n' + ', '.join([f'{group_param} = {group_el}' for group_param, group_el in zip(group_params, group)]) if group != 0 else dataset_name
            if plot_single_pivot:
                plot_pivot(d_pivot=d_pivot, step=step, legend_loc = 'upper right', logx=logx, suptitle = suptitle)

    ddd = pd.concat(dds)
    ddd_mean = pd.concat(dds_mean)
    ddd_std = pd.concat(dds_std)

    return d_pivots, ddd, ddd_mean, ddd_std, mins

def plot_best(ddd_mean, ddd_std):
    d_mean = ddd_mean.T.reorder_levels(['net', 'n_train', 'lr', None],).sort_index()
    d_std = ddd_std.T.reorder_levels(['net', 'n_train', 'lr', None],).sort_index()

    d_mins = {}
    for net_name in ['Vanilla', 'Trained']:
        n_trains = [100, 1000, 10000]
        d_min = pd.concat([d_mean.loc[pd.IndexSlice[net_name, n_train]].min(axis=0) for n_train in n_trains], axis=1)
        d_min.columns = n_trains
        d_mins[net_name] = d_min

    d_min = pd.concat(d_mins, axis = 1)


    title_dict = {
        'dataset' :{
            'sine1d' : 'Sine 1D',
            'sine2d' : 'Sine 2D',
            'flower' : 'Flower',
            'mnist' : 'MNIST',
        },
        'eps_mult' : {
            '[0, 0, 1, 0]' : r'$T(1)$',
            '[0, 0, 1, 1]' : r'$SO(2)$',
            '[0, 1, 1, 1]' : r'$SE(2)$',
        },
    }

    n_plot = len(d_min)
    # fig, axs = plt.subplots(nrows=n_plot, figsize=(3, 3*n_plot))
    fig, axs = plt.subplots(ncols=n_plot, figsize=(2.5*n_plot, 2.5), tight_layout = True, sharex=True, sharey=False)

    for ax, (index, row) in zip(axs, d_min.iterrows()):

        dataset, eps_mult, noise_std = index

        for net_name in d_min.columns.get_level_values(0).unique():

            title = title_dict['dataset'][dataset] + ', ' + title_dict['eps_mult'][eps_mult]
            row[net_name].plot(marker = 'o', logx=True, logy=True, ax = ax, label = net_name, title = title)
    axs[0].legend()
    axs[0].set_ylabel('Test loss')
    fig.supxlabel('Train size')
    plt.show()

def return_table(df: pd.DataFrame, step: int, group_params: List[str], hyper_params: List[str]):

    d = df.copy()

    if step == 1:
        d = d[~d['tags'].astype(str).str.contains(f'predict')].reset_index(drop=True)
    if step == 2:
        d = d[d['tags'].astype(str).str.contains('predict')].reset_index(drop=True)

    d = stringify_dict(d, group_params)

    d = pivot(d, index = ['data_dir'] + group_params , columns = hyper_params +  ['seed'], values = 'test_loss')

    if step == 2:
        d = d.drop('Predict-CalculatedP', axis = 1, level = 3)
    return d

def plot_vertical(d_pivots):
    raise DeprecationWarning
    n_plots = len(d_pivots)
    plot_dir = 'x'

    (n_height, n_width) = (1, n_plots) if plot_dir == 'x' else (n_plots, 1)
    fig, axs = plt.subplots(n_plots, 1, figsize=(5*n_height, 2*n_width), tight_layout=True)

    for i, ((dataset_name, group), d_pivot_content) in enumerate(d_pivots.items()):
        d_pivot = d_pivot_content['d_pivot']
        ax = axs[i]

        unstack = lambda d, metric: d.apply(metric).unstack().reset_index(level=0, drop = True)
        d_mean, d_std = unstack(d_pivot, pd.Series.mean), unstack(d_pivot, pd.Series.std)

        d_mean.plot(kind='barh', xerr=d_std, ax = ax, legend = False)
        ax.set_title(f'{dataset_name} - {group}')
        if i == 0: ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.5), ncol = 2)
        
        # if dataset_name in default_log: ax.set_xscale('log')







def rename_net(d_pivot, level = 3, index = 0):
    renames = {
        'Predict-NoneP': 'Vanilla', 
        'Predict-TrainedP': 'Trained',
        'Predict-CalculatedP': 'Pre-calculated', 
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