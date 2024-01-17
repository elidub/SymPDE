
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from data.utils import d_to_LT

def plot_pred(input, output, dx, dt, x_start, x_end, y_start, y_end):
    L, T = d_to_LT(input, dx, dt)
    data = np.stack([input, output, np.abs(input - output)])
    titles = ['Ground-truth', 'Prediction', 'Absolute Error']

    # Take the same vmin and vmax for all plots with some margin 
    # because the prediction can be outside the range of the data
    margin = 1.1
    vmin, vmax = input.min()*margin, input.max()*margin

    vmins, vmaxs = [vmin, vmin, 0], [vmax, vmax, np.nanmax(data[-1])]
    cmaps = ['PuOr_r', 'PuOr_r', 'gray']

    fig, axs = plt.subplots(1, len(data), figsize=(8, 4), sharex=True, sharey=True)

    for ax, d, title, vmin, vmax, cmap in zip(axs, data, titles, vmins, vmaxs, cmaps):
        img = ax.imshow(d, origin = 'lower', extent=[0,L,0,T], cmap=cmap, aspect='auto',
                vmin = vmin, vmax = vmax
            )
        fig.colorbar(img, ax=ax, location='bottom', shrink = 0.9, pad = 0.08, aspect = 20)

        # ax.axline((0, y_start), (L, y_start), color='black', linestyle = '--', linewidth=2)

        ax.set_title(title)

    label_pos_x = -0.15
    axs[0].text(label_pos_x, np.mean([x_start, x_end])/(y_end - x_start), 'Input', rotation = 'vertical', transform=axs[0].transAxes, horizontalalignment='center', verticalalignment='center')
    axs[0].text(label_pos_x, np.mean([y_start, y_end])/(y_end - x_start), 'Output', rotation = 'vertical', transform=axs[0].transAxes, horizontalalignment='center', verticalalignment='center')

    fig.supxlabel('x', y = 0.1)
    fig.supylabel('t')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def plot_1d(t, L, T):
    plt.figure(figsize=(12,8))
    plt.imshow(t, origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
    plt.colorbar()
    plt.xlabel('x [m]', fontsize=34)
    plt.ylabel('t [s]', fontsize=34)
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    plt.show()

def plot_1ds(us, dxs, dts, nrows = None, ncols = None, vminmax = False, title = None, l = 1, figsize = (12,8)):

    if nrows is None:
        nrows = len(us)
    if ncols is None:
        ncols = 1

    fig, axs = plt.subplots(nrows, ncols, figsize=np.array([figsize[0]*ncols*l,figsize[1]*nrows*l]), sharex=False, sharey=False, constrained_layout=True)

    if vminmax:
        vmin, vmax = us.min(), us.max()
    else:
        vmin, vmax = None, None

    axs = axs.flatten() if int(nrows*ncols)!=1 else [axs]

    for i, (u, dx, dt, ax) in enumerate(zip(us, dxs, dts, axs)):
        L, T = d_to_LT(u, dx, dt)
        im = ax.imshow(u, origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto', vmin = vmin, vmax = vmax)
        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')
        # ax.set_axis_off()

        # if i == 0:
        #     ax.set_ylabel('SymPDE')#, fontsize=34)
        # if i == 3:
        #     ax.set_ylabel('Mathematica')#, fontsize=34)

        # fig.colorbar(im, ax=ax, shrink = 0.6, pad = 0.08, aspect = 20, location='bottom')

    fig.supxlabel('x')
    fig.supylabel('t')
    fig.suptitle(title)

    # fig_dir = '../assets/figs/mathematica_data_comparison/'
    # save_title = title.split(':')[0]
    # fig_file = os.path.join(fig_dir, save_title + '.png')
    # fig.savefig(fig_file, dpi = 300)


    plt.show()


def plot_1d_dict(u_dict):
    nrows, ncols = len(u_dict), len(list(u_dict.values())[0][0])
    figsize = [12, 8]
    figsize = [3, 4]
    fig, axs = plt.subplots(nrows, ncols, figsize=np.array([figsize[0]*ncols,figsize[1]*nrows]), sharex=True, sharey=True, constrained_layout=True)

    for i, (pde_name, d) in enumerate(u_dict.items()):
        us, dx, dt = d
        L, T = d_to_LT(us, dx, dt)
        for j in range(ncols):
            ax = axs[i, j] if nrows > 1 and ncols > 1 else (axs[i] if nrows > 1 else axs[j] if ncols > 1 else None)
            print(us.shape, L, T)
            ax.imshow(us[j], origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)
            ax.set_title(pde_name, fontsize=34) if j == 1 else None
            ax.set_axis_off()


    fig.supxlabel('x', fontsize=34)
    fig.supylabel('t', fontsize=34)


    plt.show()

