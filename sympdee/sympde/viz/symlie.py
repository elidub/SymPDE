import torch
import matplotlib.pyplot as plt
import numpy as np


def wb_plot_pred(w, b, title=None,figsize = (4, 3), colorbar = True, set_axis_off = False, cmap=None):

    if b == None:
        wb = w.detach().numpy()
    else:

        b = b.view(-1, 1)

        wb = torch.cat([
            w,
            torch.full_like(b, torch.nan),
            b,
        ], dim = 1).detach().numpy()

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    im = ax.imshow(wb, cmap = cmap)
    
    ax.set_title(title)
    if set_axis_off: ax.set_axis_off()
    if colorbar: fig.colorbar(im, ax=ax)

    plt.show()
    return fig