import numpy as np
import matplotlib.pyplot as plt

from sympdesuite.coords import d_to_LT

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

    fig.supxlabel('x')
    fig.supylabel('t')
    fig.suptitle(title)

    plt.show()
