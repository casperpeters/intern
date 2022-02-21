import torch
import numpy as np


def get_lrs(mode, n_epochs, **kwargs):

    if mode == 'cyclic':
        lrs = cyclic(n_epochs, **kwargs)
    elif mode == 'geometric_decay':
        lrs = geometric_decay(n_epochs, **kwargs)
    elif mode == 'linear_decay':
        lrs = linear_decay(n_epochs, **kwargs)

    return lrs


def cyclic(n_epochs, stepsize=200, max_lr=1e-3, base_lr=1e-4):
    lrs = [base_lr]
    for epoch in range(n_epochs):
        cycle = np.floor(1 + epoch / (2 * stepsize))
        x = np.abs(epoch / stepsize - 2 * cycle + 1)
        lrs.append(base_lr + (max_lr - base_lr) * np.max(0, (1-x)))
    return lrs


def geometric_decay(n_epochs, lr_end=1e-5, start_decay=200, lr=1e-4):
    lrs = [lr]
    for epoch in range(n_epochs):
        lrs.append(lrs[epoch] * (lr_end / lrs[epoch]) ** (1 / (n_epochs - start_decay)))
    return lrs


def linear_decay():
    return


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lrs = get_lrs(mode='cyclic', n_epochs=2000)
    plt.plot(lrs)
    plt.show()
