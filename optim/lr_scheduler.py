import torch
import numpy as np


def get_lrs(mode, n_epochs, **kwargs):

    modes = ['cyclic', 'geometric_decay', 'linear_decay', 'cosine_annealing_warm_restarts']

    if mode == 'cyclic':
        lrs = cyclic(n_epochs, **kwargs)
    elif mode == 'geometric_decay':
        lrs = geometric_decay(n_epochs, **kwargs)
    elif mode == 'linear_decay':
        lrs = linear_decay(n_epochs, **kwargs)
    elif mode == 'cosine_annealing_warm_restarts':
        lrs = cosine_annealing_warm_restarts(n_epochs, **kwargs)
    else:
        raise ValueError('mode not recognized, use one of these: ' + modes)

    return lrs


def plot_lrs(lrs, y_log=False):
    plt.plot(lrs)
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('learning rate', fontsize=18)
    plt.title('learning rate over epochs', fontsize=20)
    if y_log:
        plt.yscale('log')
    return plt.gca()


def cyclic(n_epochs, step_size=200, max_lr=1e-3, base_lr=1e-4):
    lrs = [base_lr]
    for epoch in range(n_epochs):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lrs.append(base_lr + (max_lr - base_lr) * np.max([0, (1-x)]))
    return lrs


def geometric_decay(n_epochs, lr_end=1e-5, start_decay=200, lr=1e-4):
    lrs = [lr]
    for epoch in range(n_epochs):
        lrs.append(lrs[epoch] * (lr_end / lrs[epoch]) ** (1 / (n_epochs - start_decay)))
    return lrs


def cosine_annealing_warm_restarts(n_epochs, max_lr=1e-2, min_lr=1e-3, T_i=200, T_mult=1, lr_decay=.7):
    lrs = [(max_lr + min_lr) / 2]
    for epoch in range(n_epochs):
        T_cur = epoch % T_i
        if T_cur == 0 and epoch != 0 and lr_decay is not None:
            min_lr *= lr_decay
            max_lr *= lr_decay
            T_i *= T_mult
        lrs.append(min_lr + .5 * (max_lr - min_lr) * (1 + np.cos(np.pi * T_cur / T_i)) / 2)
    return lrs


def linear_decay():
    raise ValueError('not implemented yet')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lrs = get_lrs(mode='cosine_annealing_warm_restarts', n_epochs=2000)
    plot_lrs(lrs)
    print(np.min(lrs))
