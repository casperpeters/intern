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
    elif mode == 'cyclic_annealing':
        lrs = cyclic_annealing(n_epochs, stepsize=200, lr_end=1e-4, start_decay=200, lr=1e-3)
    else:
        raise ValueError('mode not recognized, use one of these: ' + modes)

    return lrs


def cyclic(n_epochs, stepsize=200, max_lr=1e-3, base_lr=1e-4):
    lrs = [base_lr]
    for epoch in range(n_epochs):
        cycle = np.floor(1 + epoch / (2 * stepsize))
        x = np.abs(epoch / stepsize - 2 * cycle + 1)
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


def cyclic_annealing(n_epochs, stepsize=200, lr_end=1e-4, start_decay=200, lr=1e-3):

    lrs_cyclic = np.array(cyclic(n_epochs, stepsize=stepsize, max_lr=lr, base_lr=lr_end))
    lrs_geo = np.array(geometric_decay(n_epochs, lr_end=lr_end, start_decay=start_decay, lr=lr))
    lrs_base = np.ones(n_epochs+1) * lr_end

    peaks = np.where((lrs_cyclic[1:-1] > lrs_cyclic[0:-2]) * (lrs_cyclic[1:-1] > lrs_cyclic[2:]))[0] + 1
    dips = np.zeros(len(peaks)+1).astype(int)
    for i in range(len(peaks)+1):
        dips[i] = int(i*2*stepsize)

    for i in range(dips.shape[0]-1):
        lrs_base[dips[i]:dips[i+1]] = np.array(cyclic(stepsize*2-1 , stepsize=stepsize, max_lr=lrs_geo[peaks[i]], base_lr=lr_end))

    return lrs_base

def linear_decay():
    raise ValueError('not implemented yet')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lrs = get_lrs(mode='cyclic_annealing', n_epochs=2000)
    plt.plot(lrs)
    plt.show()
