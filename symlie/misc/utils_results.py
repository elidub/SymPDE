import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from data.generate_2d import Create2dData


def plot_data(dataset, N_plot=5):
    for data_kwargs in ['data_kwargs_show', 'data_kwargs']:
        create_data = Create2dData(dataset['create_sample_func'], dataset[data_kwargs], dataset['transform_kwargs'])
        out = create_data(N = N_plot)
        x, y = out['x'].reshape(N_plot, *dataset[data_kwargs]['grid_size']), out['y']
        dataset['plot_func'](x, y)

def assert_unique(df_map_new):
    df_map_new = df_map_new.copy()
    del df_map_new['run_id']
    for col in df_map_new.columns:
        df_map_new[col] = df_map_new[col].astype(str)
    assert len(df_map_new) == len(df_map_new.drop_duplicates())

def pivot(d, columns, index = 'seed', values = 'test_loss'):
    if type(columns) == str: columns = [columns]
    if type(values) == str:  values = [values]
    if type(index) == str:   index = [index]

    d = d[columns + index + values].sort_values(by= columns + index).reset_index(drop=True).pivot(index=index, columns=columns, values=values)
    return d

def plot_pivot(d_pivot, figsize=(4, 4), logx = False, legend_loc = None):

    unstack = lambda d, metric: d.apply(metric).unstack().reset_index(level=0, drop = True)

    d_mean, d_std = unstack(d_pivot, pd.Series.mean), unstack(d_pivot, pd.Series.std)

    fig, ax = plt.subplots(figsize=figsize)
    d_mean.plot(kind='barh', xerr=d_std, ax = ax)
    
    if logx: ax.set_xscale('log')
    
    title = d_pivot.columns[0][0]
    if title == 'test_loss': title = 'Test Loss'
    
    ax.set_xlabel(title)
    ax.legend(loc=legend_loc)

    plt.show()