from matplotlib import pyplot as plt

def plot2d(x, y = None, l = 1, set_axis_off = True):
    N_plot = len(x)
    fig, axs = plt.subplots(1, N_plot, figsize = (N_plot*l, l), sharey=True, sharex=True)
    if N_plot == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(x[i], aspect='auto')
        if set_axis_off: ax.set_axis_off()
        if y is not None: axs[i].set_title(f'y = {y[i]}')
    plt.show()


def plot1d(x, y):
    x = x.squeeze(1)

    normalize = lambda value, min_val, max_val: (value - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap('viridis', 2)
    for x_i, y_i in zip(x, y):
        val = normalize(y_i, y.min(), y.max())
        plt.plot(x_i, color = cmap(val), alpha = 0.5, linestyle = '-')
    plt.show()

