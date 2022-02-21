import torch
import numpy as np


def get_lrs(mode, n_epochs, **kwargs):
    """
    This function returns a list of learning rates for a given number of epochs.
    It can be used for cyclical learning rates, exponential learning rates, or
    linear learning rates.

    Parameters
    ----------
    mode : str
        The type of learning rate schedule.
        Must be one of the following:
        'cyclic' : A basic cyclical learning rate, where the learning rate
            increases linearly from an initial value to some maximum value, and
            then decreases linearly to some minimum value.
        'geometric_decay' : A learning rate schedule where the learning rate
            starts at some value, and then multiplicatively decreases by some factor
            every epoch.
        'linear_decay' : A learning rate schedule where the learning rate starts
            at some value, and then decreases linearly by some factor every epoch.
        'cosine_annealing_warm_restarts' : A learning rate schedule where the
            learning rate starts at 1, and then decreases following a cosine
            curve. The cosine schedule repeats for some number of cycles, after
            which the learning rate decreases more slowly.
        'cyclic_annealing' : A learning rate schedule where the learning rate
            starts at 1, and then decreases linearly. The schedule then
            alternates between increasing linearly and decreasing linearly,
            starting with the latter.

    n_epochs : int
        The number of epochs to train for.

    kwargs : dict
        Keyword arguments for the different learning rate schedules.

    Returns
    -------
    lrs : list
        A list of learning rates for each epoch.
    """

    modes = ['cyclic', 'geometric_decay', 'linear_decay', 'cosine_annealing_warm_restarts', 'cyclic_annealing']

    if mode == 'cyclic':
        lrs = cyclic(n_epochs, **kwargs)
    elif mode == 'geometric_decay':
        lrs = geometric_decay(n_epochs, **kwargs)
    elif mode == 'linear_decay':
        lrs = linear_decay(n_epochs, **kwargs)
    elif mode == 'cosine_annealing_warm_restarts':
        lrs = cosine_annealing_warm_restarts(n_epochs, **kwargs)
    elif mode == 'cyclic_annealing':
        lrs = cyclic_annealing(n_epochs, **kwargs)
    else:
        raise ValueError('mode not recognized, use one of these: ' + ', '.join(map(str, modes)))

    return lrs


def cyclic(n_epochs, stepsize=200, max_lr=1e-3, base_lr=1e-4):
    """
    This is a function to create a cyclic learning rate schedule.

    Parameters
    ----------
    n_epochs : int
        The number of epochs for which the schedule is created.
    stepsize : int, optional
        The stepsize for the schedule. This is the period over which the
        learning rate is constant.
    max_lr : float, optional
        The maximum learning rate.
    base_lr : float, optional
        The minimum learning rate.

    Returns
    -------
    lrs : list
        The learning rates for each epoch.

    References
    ----------
    https://arxiv.org/pdf/1506.01186.pdf
    """

    lrs = [base_lr]
    for epoch in range(n_epochs):
        cycle = np.floor(1 + epoch / (2 * stepsize))
        x = np.abs(epoch / stepsize - 2 * cycle + 1)
        lrs.append(base_lr + (max_lr - base_lr) * np.max([0, (1-x)]))
    return lrs


def geometric_decay(n_epochs, lr_end=1e-5, start_decay=200, lr=1e-3):
    """
    Generates a list of learning rates for aa geometrically decaying learning rate schedule.

    Parameters
    ----------
    n_epochs : int
        The number of epochs to train for.
    lr_end : float, optional
        The final learning rate. Default: 1e-5
    start_decay : int, optional
        The epoch to start decaying the learning rate. Default: 200
    lr : float, optional
        The initial learning rate. Default: 1e-4

    Returns
    -------
    lrs : list
        The learning rates for each epoch.
    """

    lrs = [lr]
    for epoch in range(n_epochs):
        lrs.append(lrs[epoch] * (lr_end / lrs[epoch]) ** (1 / (n_epochs - start_decay)))
    return lrs


def cosine_annealing_warm_restarts(n_epochs, max_lr=2e-3, min_lr=4e-4, T_i=200, T_mult=1, lr_decay=.85):
    """
    Cosine annealing with warm restarts, described in paper []

    Parameters
    ----------
    n_epochs : int
        The number of epochs to run for.
    max_lr : float, optional
        The maximum learning rate to use.
    min_lr : float, optional
        The minimum learning rate to use.
    T_i : int, optional
        The number of epochs to run at max_lr before annealing.
    T_mult : int, optional
        The factor to increase T_i by after each restart.
    lr_decay : float, optional
        The factor to decrease max_lr and min_lr by.

    Returns
    -------
    lrs : list
        The learning rates for each epoch.

    References
    ----------
    "SGDR: stochastic gradient descent with warm restarts"
    https://arxiv.org/abs/1608.03983
    """

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
        lrs_base[dips[i]:dips[i+1]] = np.array(cyclic(stepsize*2-1, stepsize=stepsize, max_lr=lrs_geo[peaks[i]], base_lr=lr_end))

    return lrs_base.tolist()


def linear_decay(n_epochs, lr_start=1e-3, lr_stop=1e-5):
    """
    Creates a list of learning rates for a linear decay schedule.

    Parameters
    ----------
    n_epochs : int
        The number of epochs for which the model will be trained.
    lr_start : float, optional
        The learning rate at the start of the schedule.
    lr_stop : float, optional
        The learning rate at the end of the schedule.

    Returns
    -------
    lrs : list
        The learning rates for each epoch.
    """

    lrs = np.linspace(lr_start, lr_stop, n_epochs)
    return lrs.tolist()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lrs = get_lrs(mode='linear_decay', n_epochs=2000)
    plt.plot(lrs)
    plt.show()
