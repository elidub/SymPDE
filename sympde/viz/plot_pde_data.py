
import matplotlib.pyplot as plt
import numpy as np

from data.utils import d_to_LT


def plot_1d(t, L, T):
    plt.figure(figsize=(12,8))
    plt.imshow(t, origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
    plt.colorbar()
    plt.xlabel('x [m]', fontsize=34)
    plt.ylabel('t [s]', fontsize=34)
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    plt.show()

def plot_1ds(us, L, T):
    n_rows, n_cols = us.shape[:2]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=np.array([12*n_cols,8*n_rows]), sharex=True, sharey=True, constrained_layout=True)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i,j]
            ax.imshow(us[i,j], origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)

    fig.supxlabel('x', fontsize=34)
    fig.supylabel('t', fontsize=34)


    plt.show()


def plot_1d_dict(u_dict):
    n_rows, n_cols = len(u_dict), len(list(u_dict.values())[0][0])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=np.array([12*n_cols,8*n_rows]), sharex=True, sharey=True, constrained_layout=True)

    for i, (pde_name, d) in enumerate(u_dict.items()):
        us, dx, dt = d
        L, T = d_to_LT(us, dx, dt)
        for j in range(n_cols):
            ax = axs[i,j] if n_rows > 1 else axs[j]
            ax.imshow(us[j], origin = 'lower', extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.tick_params(axis='both', which='minor', labelsize=28)
            ax.set_title(pde_name, fontsize=34)


    fig.supxlabel('x', fontsize=34)
    fig.supylabel('t', fontsize=34)


    plt.show()

