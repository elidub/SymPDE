import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import os

def imshows(plots, titles = None, vminmaxs = None, suptitle = None, colorbar = False, axis_off = False, plot_labels = False, l =1, imshow_kwargs = {}, rows = True, text_vals = None):
    n_plots = len(plots)

    (nrows, ncols) = (1, n_plots) if rows else (n_plots, 1)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*nrows*l, 3*ncols*l), tight_layout=True)

    vminmax_absmax = np.abs([np.abs(plot).max() for plot in plots]).max()

    if not isinstance(text_vals, list):
        text_vals = [text_vals] * n_plots

    for i in range(n_plots):
        ax = axs if n_plots == 1 else axs[i]

        if vminmaxs == 'single':
            vmin = -vminmax_absmax
            vmax = vminmax_absmax
        elif vminmaxs == 'separate':
            vmin = -np.abs(plots[i]).max()
            vmax = np.abs(plots[i]).max()
            print(vmin, vmax)
        elif isinstance(vminmaxs, list) or isinstance(vminmaxs, tuple) or isinstance(vminmaxs, np.ndarray):
            vmin = vminmaxs[i][0]
            vmax = vminmaxs[i][1]
        else:
            vmin, vmax = None, None

        im = ax.imshow(plots[i], vmin=vmin,vmax=vmax,**imshow_kwargs[i] if isinstance(imshow_kwargs, list) else imshow_kwargs)

        if plot_labels:
            for (j_label, i_label), label in np.ndenumerate(plots[i]):
                ax.text(i_label, j_label, f'{label:.4f}', ha='center', va='center', fontsize=16)

        ax.set_title(titles[i] if titles is not None else None)

        if text_vals[i]:
            plot_vals(plots[i], ax)

        if axis_off:
            ax.set_axis_off()

        if colorbar:
            if n_plots > 1:
                # put colorbar next to axis
                divider = make_axes_locatable(axs[i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)
            else:
                fig.colorbar(im, ax=ax)
    
    if suptitle is not None:
        # fig.suptitle(suptitle, y = .85, fontsize=16)
        fig.suptitle(suptitle, y = 1., fontsize=16)


    plt.show()

def plot_vals(x, ax):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ax.text(j, i, f'{x[i, j]:.2f}', ha='center', va='center', color='white')

def simple_imshow(x, l = 1, print_values = False, precision = 2, title = None, imshow_kwargs = {}):
    fig, ax = plt.subplots(1, 1, figsize=np.array(x.T.shape)*l, tight_layout=True)
    ax.imshow(x, **imshow_kwargs)
    ax.set_title(title)

    if print_values:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ax.text(j, i, f'{x[i, j]:.{precision}f}', ha='center', va='center', color='white')

    ax.axis('off')
    plt.show()
    return fig

def imshow(x, figsize = (3,3), l = 1):
    plt.figure(figsize = np.array(figsize)*l)
    plt.imshow(x)
    plt.show()

def savefig(fig, name, subdir = '', path = '/Users/elias/EliasMBA/Projects/Uni/Thesis/ai_thesis/figures/code', tight_layout = True, **kwargs):
    if not hasattr(kwargs, 'dpi'): kwargs['dpi'] = fig.dpi
    os.makedirs(os.path.join(path, subdir), exist_ok = True)
    fig.savefig(os.path.join(path, subdir, f'{name}.png'), **kwargs)