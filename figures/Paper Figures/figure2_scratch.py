import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.neighbors import KernelDensity


def shrink(data, rows, cols):
    return data.reshape(rows, data.shape[0]//rows, cols, data.shape[1]//cols).sum(axis=1).sum(axis=2)


if __name__ == '__main__':
    temporal_connections = 10 * torch.tensor([
        [0, 1, 0],
        [-1, 0, 1],
        [0, -1, 0]
    ])
    n_h, n_v, T = 3, 30, 100
    bandwidth = 1
    res = 10
    firing_rate = 0.1
    threshold = 0.01

    t = np.arange(start=0, stop=T*res)
    population_waves = np.empty((n_h, T*res))
    mother_trains = np.empty((n_h, T*res))
    neuron_indexes = torch.arange(end=n_v).view(n_h, n_v // n_h)
    data = torch.zeros(n_v, T)
    a = torch.zeros(n_v, T)
    d = torch.zeros(n_v, T)

    for i in range(n_h):
        n_peaks = np.random.randint(low=firing_rate*T // 2, high=firing_rate*T // .67)
        peak_locations = np.random.randint(low=0, high=T*res, size=n_peaks)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth*res).fit(peak_locations[:, None])
        gausfit = kde.score_samples(t[:, None])

        # create spike train based on random gaussian-peaked function
        population_waves[i, :] = np.exp(gausfit)
        mother_trains[i, :] = np.random.poisson(lam=population_waves[i, :])

    mother_trains = shrink(
        data=population_waves,
        rows=n_h,
        cols=T
    ) > threshold

    fig, axes = plt.subplots(2, 1, figsize=(5, 5), gridspec_kw={'hspace': 0.05, 'height_ratios': [8, 1]},)
    for i, pop_wave in enumerate(population_waves):
        axes[0].plot(pop_wave, label=str(i))
    axes[0].legend()
    axes[0].set_ylabel('$r_i$', fontsize=15)
    axes[0].set_xlim([0, T*res])
    axes[1].set_xlabel('time', fontsize=15)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([0, 1, 2])
    axes[1].imshow(mother_trains>0, cmap=plt.get_cmap('binary'))

    plt.savefig(r'D:\OneDrive\RU\Intern\rtrbm_master\figures\quick1.png')
    plt.show()

    for population_index in range(n_h):
        # connections to current population
        connections = temporal_connections[population_index, :]

        # delete spikes based on inhibitory connections
        inhibitory_connections = connections < 0
        if torch.sum(inhibitory_connections) > 0:
            temp = population_waves[inhibitory_connections, :]
            deletion_probabilities = -1 * torch.sum(
                input=connections[inhibitory_connections].unsqueeze(1) * shrink(
                    data=temp,
                    rows=temp.shape[0],
                    cols=T
                ),
                dim=0
            ).T
            delete_spikes = (torch.tile(
                input=torch.roll(
                    input=deletion_probabilities,
                    shifts=1,
                    dims=0
                ),
                dims=(n_v // n_h, 1)
            ) > torch.rand(n_v // n_h, T)).type(torch.float)
        else:
            delete_spikes = torch.zeros(n_v // n_h, T)

        # add spikes based on exciting connections
        exciting_connection = connections > 0
        if torch.sum(exciting_connection) > 0:
            temp = population_waves[exciting_connection, :]
            addition_probabilities = torch.sum(
                input=connections[exciting_connection].unsqueeze(1) * shrink(
                    data=temp,
                    rows=temp.shape[0],
                    cols=T
                ),
                dim=0
            ).T
            add_spikes = torch.poisson(
                torch.tile(
                    input=torch.roll(
                        input=addition_probabilities,
                        shifts=1,
                        dims=0
                    ),
                    dims=(n_v // n_h, 1)
                )
            ).type(torch.float)
        else:
            add_spikes = torch.zeros(n_v // n_h, T)

        add_spikes[add_spikes > 1] = 1

        spikes = torch.tile(
            input=torch.tensor(
                data=mother_trains[population_index, :],
                dtype=torch.float
            ),
            dims=(n_v // n_h, 1)
        ) - delete_spikes + add_spikes

        data[neuron_indexes[population_index], :] = spikes
        a[neuron_indexes[population_index], :] = add_spikes
        d[neuron_indexes[population_index], :] = delete_spikes

    x=1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    v = torch.zeros(n_v, T)
    for i in range(n_h):
        v[i*10:(i+1)*10, :] = torch.tensor(
            data=mother_trains[i, :],
        ).repeat(10, 1)

    v = np.ma.masked_where(v == 0, v)
    a = np.ma.masked_where(a == 0, a)
    d = np.ma.masked_where(d == 0, d)

    axes[0].imshow(v, cmap=plt.get_cmap('binary'), vmin=0, vmax=1)
    axes[0].imshow(a, cmap=plt.get_cmap('PRGn'), vmin=0, vmax=1.3)
    axes[0].imshow(d * v, cmap=plt.get_cmap('seismic'), vmin=0, vmax=1.3)
    axes[0].imshow(d * a, cmap=plt.get_cmap('seismic'), vmin=0, vmax=1.3)

    axes[1].imshow(data, cmap=plt.get_cmap('binary'), vmin=0, vmax=1)
    plt.savefig(r'D:\OneDrive\RU\Intern\rtrbm_master\figures\quick2.png')
    plt.show()

