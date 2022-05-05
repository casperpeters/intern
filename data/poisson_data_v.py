import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.funcs import cross_correlation
import sys


class PoissonTimeShiftedData(object):
    def __init__(
            self, neurons_per_population=10, n_populations=20, n_batches=200, time_steps_per_batch=100,
            fr_mode='gaussian', delay=1, temporal_connections=None, norm=None, sparcity=0,
            **kwargs
    ):

        """
        """

        if 'frequency_range' not in kwargs:
            kwargs['frequency_range'] = [5, 10]
        if 'amplitude_range' not in kwargs:
            kwargs['amplitude_range'] = [0.4, 0.5]
        if 'phase_range' not in kwargs:
            kwargs['phase_range'] = [0, torch.pi]
        if 'lower_bound_fr' not in kwargs:
            kwargs['lower_bound_fr'] = 0
        if 'upper_bound_fr' not in kwargs:
            kwargs['upper_bound_fr'] = 3 * kwargs['amplitude_range'][1]

        if fr_mode == 'gaussian':
            if 'std_range' not in kwargs:
                kwargs['std_range'] = [1, 2]

        if norm is None:
            norm = n_populations ** 2

        # if no temporal connections are given, take half inhibitory and half exciting populations
        if temporal_connections == 'deterministic':
            if n_populations % 2 == 0:
                a = 0
            else:
                a = 1
            temporal_connections = torch.cat(
                (torch.ones(n_populations // 2 + a, n_populations),
                 torch.full(size=(n_populations // 2, n_populations), fill_value=-1)), dim=0)
            temporal_connections -= torch.diag(torch.diag(temporal_connections))

        if temporal_connections is None or temporal_connections == 'random':
            temporal_connections = self.create_random_connections(n_populations=n_populations, sparcity=sparcity)

        temporal_connections /= norm

        self.data = torch.empty(
            neurons_per_population * n_populations,
            time_steps_per_batch,
            n_batches,
            dtype=torch.float
        )

        population_waves_original = torch.zeros(n_populations, time_steps_per_batch + delay, n_batches)
        population_waves_interact = torch.zeros(n_populations, time_steps_per_batch, n_batches)
        neuron_waves_interact = torch.zeros(neurons_per_population * n_populations, time_steps_per_batch, n_batches)

        # loop over batches
        for batch_index in range(n_batches):
            # get all mother trains by looping over populations
            for h in range(n_populations):

                # get a random sine wave or gaussian as mother train firing rate
                if fr_mode == 'sine':
                    population_waves_original[h, :, batch_index] = self.get_random_sine(
                        time_steps_per_batch=time_steps_per_batch + delay, **kwargs
                    )

                elif fr_mode == 'gaussian':

                    background_wave = self.get_random_gaussian(
                        time_steps_per_batch=time_steps_per_batch + delay,
                        n=10,
                        **kwargs
                    )

                    population_waves_original[h, :, batch_index] = self.get_random_gaussian(
                        time_steps_per_batch=time_steps_per_batch + delay,
                        **kwargs
                    ) + background_wave

            # compute interactions of all populations on their resulting firing rate
            population_waves_interact_temp = population_waves_original[..., batch_index].detach().clone()
            for t in range(delay, time_steps_per_batch + delay):
                for h in range(n_populations):
                    population_waves_interact_temp[h, t] += torch.sum(
                        temporal_connections[:, h][None, :] * population_waves_interact_temp[:, t-delay]
                    ) / n_populations

                # constrain to only positive values, upper limit and remove nan
                population_waves_interact_temp[:, t] = self.constraints(population_waves_interact_temp[:, t], **kwargs)

            # cut first temporal part (not every thing has temporal connections)
            population_waves_interact[..., batch_index] = population_waves_interact_temp[:, delay:]

            for h in range(n_populations):
                neuron_waves_interact[neurons_per_population * h: neurons_per_population * (h + 1), :, batch_index] = \
                    (population_waves_interact[h, :, batch_index]).repeat(neurons_per_population, 1) * \
                    torch.linspace(0.3, 1.5, neurons_per_population)[:, None]

            # neuron_waves_interact += torch.rand()

            if torch.sum(torch.isnan(neuron_waves_interact)) > 0:
                print('nan detected')
            elif torch.sum(neuron_waves_interact < 0) > 0:
                print('zero detected')

            self.data[..., batch_index] = torch.poisson(neuron_waves_interact[..., batch_index])

        # make sure there are
        self.data[self.data < 0] = 0
        self.data[self.data > 1] = 1
        self.population_waves_original = population_waves_original
        self.population_waves_interact = population_waves_interact
        self.neuron_waves_interact = neuron_waves_interact
        self.firing_rates = torch.mean(self.data, (1, 2))
        self.delay = delay
        self.time_steps_per_batch = time_steps_per_batch
        self.temporal_connections = temporal_connections

    def get_random_gaussian(self, time_steps_per_batch, std_range, amplitude_range, n=1, **kwargs):
        T = torch.arange(time_steps_per_batch)

        def gaussian_pdf(x, mu, std):
            pdf = 1 / (torch.sqrt(torch.tensor(torch.pi)) * std) * torch.exp(-0.5 * ((x - mu) / std) ** 2)
            return pdf

        # get sum of sinusoid (more sinusoids longer periodicity)
        temp = self.get_random_sine(time_steps_per_batch, **kwargs)
        for i in range(5):
            temp += self.get_random_sine(time_steps_per_batch, **kwargs)

        # get peak locations of sum of sinusiods, let peak locations be the location of gaussian peaks
        mu = torch.where((temp[1:-1] > temp[0:-2]) * (temp[1:-1] > temp[2:]))[0] + 1
        n_samples = mu.shape[0]
        trace = torch.sum(gaussian_pdf(T[:, None], mu,
                                       torch.rand(1, n_samples) * (std_range[1] - std_range[0]) + std_range[0]), 1)

        max_fr = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        return trace / max(trace) * (max_fr / n)

    def get_random_sine(self, time_steps_per_batch, frequency_range, phase_range, amplitude_range=[0.4, 0.5], **kwargs):
        T = torch.linspace(start=0, end=2 * torch.pi, steps=time_steps_per_batch)
        amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        frequency = torch.rand(1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]
        return (amplitude * torch.sin(frequency * T - phase) + amplitude)/2

    def create_random_connections(self, n_populations, fraction_exc_inh=0.5, max_normelation=0.9, min_normelation=0.6,
                                  sparcity=0, self_conn=None):
        N_E = int(fraction_exc_inh * n_populations)
        idx = np.random.permutation(n_populations)[:N_E]
        # Weight matrix
        U = np.random.rand(n_populations, n_populations) * (max_normelation - min_normelation) + min_normelation
        U[idx, :] = - U[idx, :]

        if np.sum(U == 0) / n_populations ** 2 < sparcity:
            U.ravel()[np.random.permutation(n_populations ** 2)[:int(sparcity * n_populations ** 2)]] = 0

        if isinstance(self_conn, (float)):
            return torch.tensor(U - self_conn * np.diag(np.diag(U)))
        elif self_conn == None:
            return -torch.tensor(U - 1 * np.diag(np.diag(U)))

    def constraints(self, population_waves_interact, lower_bound_fr, upper_bound_fr, **kwargs):
        population_waves_interact[population_waves_interact < abs(lower_bound_fr)] = abs(lower_bound_fr)
        population_waves_interact[population_waves_interact > upper_bound_fr] = upper_bound_fr
        return population_waves_interact

    def add_noise(self, sigma=.001, range=.5, delete_spikes=True):
        sigma = torch.linspace(sigma-sigma*range, sigma+sigma*range, self.data.shape[0]).unsqueeze(1).unsqueeze(2)
        add = (torch.rand(self.data.shape) < sigma.tile(1, self.data.shape[1], self.data.shape[2])).type(torch.float)

        self.data += add
        if delete_spikes:
            delete = (torch.rand(self.data.shape) < sigma).type(torch.float)
            self.data -= delete
        self.data[self.data < 0] = 0
        self.data[self.data > 1] = 1
        self.firing_rates = torch.mean(self.data, (1, 2))
        return


    def plot_stats(self, T=None, batch=0, axes=None):
        if T is None or T > self.time_steps_per_batch:
            T = self.time_steps_per_batch

        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        sns.heatmap(self.data[:, :T, batch], ax=axes[0, 0], cbar=False)
        axes[0, 0].set_title('Final spikes')

        axes[0, 1].plot(self.firing_rates[torch.argsort(self.firing_rates)], '.')
        axes[0, 1].set_title('Mean firing rates (over all batches)')

        sns.heatmap(self.temporal_connections, ax=axes[0, 2], cbar=False)
        axes[0, 2].set_title('Hidden population structure')
        maxi = torch.tensor(0)
        for i, (wave_O, wave_I) in enumerate(
                zip(self.population_waves_original[..., batch], self.population_waves_interact[..., batch])):
            axes[1, 0].plot(wave_O[:T], label=str(i))
            axes[1, 1].plot(wave_I[:T], label=str(i))
            maxi = torch.max(maxi, torch.max(torch.concat([wave_O[:T], wave_I[:T]])))

        axes[1, 0].set_title('Original population waves')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_xticks([0, T + self.delay])
        axes[1, 0].set_xticklabels(['0', str(T + self.delay)])
        axes[1, 0].set_ylim([0, maxi])

        axes[1, 1].set_title('Population waves after interaction')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_xticks([0, T])
        axes[1, 1].set_xticklabels([str(self.delay), str(T + self.delay)])
        axes[1, 1].set_ylim([0, maxi])

        cross_corr = np.zeros(10)
        for i in range(10):
            cross_corr[i] = np.mean(cross_correlation(data=self.data[:, :, 0], time_shift=i, mode='Pearson'))  # mode=Correlate
        axes[1, 2].plot(cross_corr)
        axes[1, 2].set_title('Pearson cross-correlation')
        axes[1, 2].set_xlabel('Time shift')
        axes[1, 2].set_ylabel('Cross-correlation')

        return axes


if __name__ == '__main__':
    from data.reshape_data import reshape_from_batches
    from utils.funcs import pairwise_moments
    import matplotlib.pyplot as plt
    from boltzmann_machines.cp_rtrbm import RTRBM
    from tqdm import tqdm

    n_v = 100
    n_h = 10
    delay = 1  # temporal dynamics
    frequency_range = [5, 10]
    phase_range = [0, torch.pi]
    amplitude_range = [0.2, 0.5]
    sp = 1e-4

    hiddens_range = [2, 5, 8, 10, 12, 15, 20]
    N = 3  # number of RTRBMs per run

    temporal_connections = torch.randn(n_h, n_h) / n_h

    poisson = PoissonTimeShiftedData(
        neurons_per_population=n_v // n_h, n_populations=n_h, n_batches=100, time_steps_per_batch=100,
        fr_mode='gaussian', delay=delay, temporal_connections=temporal_connections, norm=1,
        frequency_range=frequency_range, amplitude_range=amplitude_range, phase_range=phase_range
    )

    poisson.plot_stats()
    plt.show()

    rtrbm = RTRBM(poisson.data, N_H=n_h, device="cpu", debug_mode=False)
    rtrbm.learn(batch_size=10, n_epochs=200, max_lr=1e-3, min_lr=8e-4, lr_schedule='geometric_decay', CDk=10, mom=0.6,
                wc=0.0002, sp=0, x=1)

    T, n_batches = poisson.data.shape[1], poisson.data.shape[2]
    vs = torch.zeros(n_v, T, n_batches)
    for batch in tqdm(range(n_batches)):
        vs[:, :, batch], _ = rtrbm.infer(poisson.data[:, :T // 2, batch], mode=1, pre_gibbs_k=100, gibbs_k=100,
                                         disable_tqdm=True)

    vs_ = reshape_from_batches(vs)
    test_ = reshape_from_batches(poisson.data)
    true_pairwise = pairwise_moments(test_, test_).flatten()
    sampled_pairwise = pairwise_moments(vs_, vs_).flatten()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].imshow(poisson.data[:, T // 2:, 0], cmap=plt.get_cmap('binary'), aspect='auto')
    axes[1].imshow(vs[:, T // 2:, 0], cmap=plt.get_cmap('binary'), aspect='auto')
    axes[2].plot(torch.mean(vs[:, T // 2:, :], (1, 2)), torch.mean(poisson.data[:, T // 2:, :], (1, 2)), '.')
    axes[3].plot(true_pairwise, sampled_pairwise, '.')

    axes[0].set_title('True (test) data', fontsize=18)
    axes[0].set_xlabel('Time', fontsize=16)
    axes[0].set_ylabel('$v$', fontsize=16)
    axes[1].set_title('Sampled data', fontsize=18)
    axes[1].set_xlabel('Time', fontsize=16)
    axes[1].set_ylabel('$v$', fontsize=16)
    axes[2].set_title('$<v_i>$', fontsize=18)
    axes[2].set_xlabel('True', fontsize=16)
    axes[2].set_ylabel('Sampled', fontsize=16)
    axes[3].set_title('$<v_iv_j>$', fontsize=18)
    axes[3].set_xlabel('True', fontsize=16)
    axes[3].set_ylabel('Sampled', fontsize=16)

    plt.tight_layout()
    plt.show()
