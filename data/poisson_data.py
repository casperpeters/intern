import torch


def poisson_data(n_batches=50, n_pop=5, n_per_pop=20, fr=.1, T=100, correlation=.8, snr=20):
    data = torch.empty(n_pop * n_per_pop, T, n_batches)
    for batch in range(n_batches):
        data[..., batch] = poisson_data_batch(n_pop, n_per_pop, fr, T, correlation, snr)
    return data


def poisson_data_batch(n_pop, n_per_pop, fr, T, correlation, snr):
    for i in range(n_pop):
        mother_train = constant_mother_train(fr / correlation, T)

        for n in range(n_per_pop):
            spikes = mother_train.expand(n_per_pop, T)
            spikes = spikes - spikes * (torch.rand(n_per_pop, T) > correlation)
            spikes = spikes + (spikes == 0) * (torch.rand(n_per_pop, T) < (correlation / snr))

            spikes[spikes > 1] = 1
        if i == 0:
            data = spikes.clone().detach()
        else:
            data = torch.cat((data, spikes), dim=0)
    return data


def constant_mother_train(fr, steps):
    rates = fr * torch.ones(steps)
    mother_train = torch.poisson(rates)
    return mother_train
