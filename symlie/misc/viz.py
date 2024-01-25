from matplotlib import pyplot as plt

def imshow(x, y = None, l = 1, set_axis_off = True):
    N_plot = len(x)
    fig, axs = plt.subplots(1, N_plot, figsize = (N_plot*l, l), sharey=True, sharex=True)
    if N_plot == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(x[i], aspect='auto')
        if set_axis_off: ax.set_axis_off()
        if y is not None: axs[i].set_title(f'y = {y[i]}')
    plt.show()