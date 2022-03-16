"""
Always give an ax as input and return it.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.funcs import get_param_history


def mean_std_parameters_history(parameter_history, axes=None):
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    W, U, b_H, b_V, b_init = get_param_history(parameter_history)
    epochs = np.arange(W.shape[0])
    params = [W, U, b_H, b_V, b_init]
    for param in params:
        axes[0].plot(epochs, torch.mean(param, (1, 2)))
        axes[1].plot(epochs, torch.std(param, (1, 2)))
    axes[0].legend(['W', 'U', 'b_H', 'b_V', 'b_init'])
    axes[0].set_ylabel('Means', fontsize=15)
    axes[1].set_ylabel('STDs', fontsize=15)
    axes[1].set_xlabel('epochs', fontsize=15)
    plt.tight_layout()

    return axes
