import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np

def imshows(plots, titles = None, vminmaxs = None, suptitle = None, imshow_kwargs = {}):
    n_plots = len(plots)

    fig, axs = plt.subplots(1, n_plots, figsize=(3*n_plots, 4), tight_layout=True)

    vminmax_absmax = np.abs(plots).max()

    for i in range(n_plots):
        ax = axs[i]

        if vminmaxs == 'single':
            vmin = -vminmax_absmax
            vmax = vminmax_absmax
        elif vminmaxs == 'separate':
            vmin = -np.abs(plots[i]).max()
            vmax = np.abs(plots[i]).max()
        elif isinstance(vminmaxs, list) or isinstance(vminmaxs, tuple) or isinstance(vminmaxs, np.ndarray):
            vmin = vminmaxs[i][0]
            vmax = vminmaxs[i][1]
        else:
            vmin, vmax = None, None

        im = ax.imshow(plots[i], vmin=vmin,vmax=vmax,**imshow_kwargs[i] if isinstance(imshow_kwargs, list) else imshow_kwargs)

        ax.set_title(titles[i] if titles is not None else None)


        # put colorbar next to axis
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    
    if suptitle is not None:
        fig.suptitle(suptitle, y = .85, fontsize=16)


    plt.show()