
from pde import plot_kymograph
import matplotlib.pyplot as plt

# def plot_1d(storage, data):
#     assert storage.data.shape == data.shape
#     storage.data = data
#     plot_kymograph(storage)


def plot_1d(t, L, T):
    plt.figure(figsize=(12,8))
    plt.imshow(t, extent=[0,L,0,T], cmap='PuOr_r', aspect='auto')
    plt.colorbar()
    plt.title('KdV (pseudospectral)', fontsize=36)
    plt.xlabel('x [m]', fontsize=34)
    plt.ylabel('t [s]', fontsize=34)
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    plt.show()