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


def plot1d(x, y, l=1):
    x = x.squeeze(1)

    normalize = lambda value, min_val, max_val: (value - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=np.array([2, 1])*5*l)
    cmap = plt.cm.get_cmap('viridis', 2)
    for x_i, y_i in zip(x, y):
        val = normalize(y_i, y.min(), y.max())
        plt.plot(x_i, color = cmap(val), alpha = 0.5, linestyle = '-')
    plt.show()

