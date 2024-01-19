import numpy as np
from matplotlib import pyplot as plt

def create_flat(N: int, space_length: int, noise_std: float, y_low: int, y_high: int) -> tuple:
    """Create a sample from the flat dataset.

    Args:
        N: Number of samples to create.
        space_length: Length of the space.
        noise_std: Standard deviation of the noise.
        y_low: Lower bound of the y values.
        y_high: Upper bound of the y values.

    Returns:
        x: Array of shape (N, 1) with the x values.
        y: Array of shape (N, 1) with the y values.
    """
    x = np.linspace(0, 1, space_length).reshape(-1, 1)
    k = np.random.randint(y_low, y_high, size = (N,))
    # k = np.arange(y_low, y_high)
    # N = len(k)
    shift = np.random.randint(0, space_length, size = (N,))
    noise = np.random.normal(0, noise_std, size = (N, space_length))

    sins = np.sin(2*np.pi*x*k - shift).T

    sins_noised = sins + noise
    return {'x': sins_noised, 'y':k}


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)




def plot_flat(x, y, N_plot = 50):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap('viridis', 2)
    for i in range(N_plot):
        val = normalize(y[i], y.min(), y.max())
        plt.plot(x[i], color = cmap(val), alpha = 0.5, linestyle = '-')
    plt.show()