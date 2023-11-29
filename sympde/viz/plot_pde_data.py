
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_1ds(us, dx, dt, vminmax = False):
    n_rows, n_cols = us.shape[:2]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=np.array([12*n_cols,8*n_rows]), sharex=True, sharey=True, constrained_layout=True)

    if vminmax:
        vmin, vmax = us.min(), us.max()
    else:
        vmin, vmax = None, None

    for i in range(n_rows):
        for j in range(n_cols):
            L, T = d_to_LT(us[i,j], dx, dt)
            ax = axs[i,j] if n_rows > 1 else axs[j] if n_cols > 1 else axs
            ax.imshow(us[i,j], origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto', vmin = vmin, vmax = vmax)
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)

    fig.supxlabel('x', fontsize=34)
    fig.supylabel('t', fontsize=34)


    plt.show()


def plot_1d_dict(u_dict):
    n_rows, n_cols = len(u_dict), len(list(u_dict.values())[0][0])
    figsize = [12, 8]
    figsize = [3, 4]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=np.array([figsize[0]*n_cols,figsize[1]*n_rows]), sharex=True, sharey=True, constrained_layout=True)

    for i, (pde_name, d) in enumerate(u_dict.items()):
        us, dx, dt = d
        L, T = d_to_LT(us, dx, dt)
        for j in range(n_cols):
            ax = axs[i, j] if n_rows > 1 and n_cols > 1 else (axs[i] if n_rows > 1 else axs[j] if n_cols > 1 else None)
            ax.imshow(us[j], origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)
            ax.set_title(pde_name, fontsize=34) if j == 1 else None
            ax.set_axis_off()


    fig.supxlabel('x', fontsize=34)
    fig.supylabel('t', fontsize=34)


    plt.show()

