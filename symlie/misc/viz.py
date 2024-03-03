from matplotlib import pyplot as plt
import numpy as np

def plot2d(x, y = None, l = 1, set_axis_off = True, max_grid = None):
    N_plot = len(x)
    fig, axs = plt.subplots(1, N_plot, figsize = (N_plot*l, l), sharey=True, sharex=True)
    if N_plot == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        x_i = x[i] if max_grid is None else x[i][:max_grid, :max_grid]
        ax.imshow(x_i, aspect='auto')
        if set_axis_off: ax.set_axis_off()
        if y is not None: axs[i].set_title(f'y = {y[i]}')
    plt.show()


def plot1d(x, y = None, l=1):
    x = x.squeeze(1)


    n_y = y.shape[1] if len(y.shape) > 1 else 1
    y_keys = ['k', 'A'][:n_y]

    fig, ax = plt.subplots(figsize=np.array([2, 1])*5*l)
    ax.plot(x.T)
    ax.legend(np.round(y, 2), title = ', '.join(y_keys))
    plt.show()    

    return


    y_keys, y_vals = y.keys(), np.array(tuple(y.values())).T

    fig, ax = plt.subplots(figsize=np.array([2, 1])*5*l)
    ax.plot(x.T)
    ax.legend(np.round(y_vals, 2), title = ', '.join(y_keys))
    plt.show()
    
    return


    if y is None:
        y = np.ones(len(x))
    assert len(x) == len(y)

    normalize = lambda value, min_val, max_val: (value - min_val) / (max_val - min_val) if max_val - min_val > 0 else 0

    fig, ax = plt.subplots(figsize=np.array([2, 1])*5*l)
    cmap = plt.cm.get_cmap('viridis', 2)
    for x_i, y_i in zip(x, y):
        val = normalize(y_i, y.min(), y.max())
        plt.plot(x_i, color = cmap(val), alpha = 0.5)
    plt.show()

