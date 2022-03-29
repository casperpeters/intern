import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

class PoissonTimeShiftedData(object):
    def __init__(
            self,
            neurons_per_population=10,
            n_populations=20,
            n_batches=200,
            duration=0.1, dt=1e-2,
            fr_mode='gaussian', delay=1, temporal_connections='random', corr=None, show_connection=True,
            **kwargs

    ):

        """
        """

        if fr_mode == 'sine':
            if 'frequency_range' not in kwargs:
                kwargs['frequency_range'] = [40, 200]
            if 'amplitude_range' not in kwargs:
                kwargs['amplitude_range'] = [40, 80]
            if 'phase_range' not in kwargs:
                kwargs['phase_range'] = [0, torch.pi]

        if fr_mode == 'gaussian':
            if 'fr_range' not in kwargs:
                kwargs['fr_range'] = [5, 40]
            if 'mu_range' not in kwargs:
                kwargs['mu_range'] = [0, duration]  # in seconds
            if 'std_range' not in kwargs:
                kwargs['std_range'] = [2 * dt, 10 * dt]  # in seconds
            if 'n_range' not in kwargs:
                kwargs['n_range'] = [0.1, 0.8]  # average number of different gaussians per bin
            if 'lower_bound_fr' not in kwargs:
                kwargs['lower_bound_fr'] = 0.05
            if 'upper_bound_fr' not in kwargs:
                kwargs['upper_bound_fr'] = 1.2 * kwargs['fr_range'][1]


        if corr == None:
            corr = n_populations ** 2

        # if no temporal connections are given, take half inhibitory and half exciting populations
        if temporal_connections is None:
            temporal_connections = torch.cat(
                (torch.ones(n_populations, n_populations // 2),
                 torch.full(size=(n_populations, n_populations // 2), fill_value=-1)),
                dim=1
            ) / corr

        if temporal_connections == 'random':
            temporal_connections = self.create_random_connections(n_populations=n_populations,
                                                                  show_connection=show_connection) / corr


        # initialize empty tensors
        time_steps_per_batch = int(duration / dt)

        self.data = torch.empty(
            neurons_per_population * n_populations,
            time_steps_per_batch,
            n_batches,
            dtype=torch.float
        )

        population_waves_original = torch.zeros(n_populations, time_steps_per_batch+delay, n_batches)
        population_waves_interact = torch.zeros(n_populations, time_steps_per_batch, n_batches)
        neuron_waves_interact = torch.zeros(neurons_per_population * n_populations, time_steps_per_batch, n_batches)

        # loop over batches
        for batch_index in range(n_batches):

            # get all mother trains by looping over populations
            for population_index in range(n_populations):
                # get a random sine wave as mother train firing rate
                if fr_mode == 'sine':
                    population_waves_original[population_index, :, batch_index] = self.get_random_sine(T=duration+delay*dt, dt=dt, **kwargs)
                elif fr_mode == 'gaussian':
                    population_waves_original[population_index, :, batch_index] = self.get_random_gaussian(T=duration+delay*dt, dt=dt, **kwargs) + kwargs['lower_bound_fr']

                population_waves_interact[population_index, :, batch_index] = population_waves_original[population_index, delay:, batch_index] + torch.sum(
                    temporal_connections[:, population_index][None, :].repeat(time_steps_per_batch, 1).T * population_waves_original[:, :-delay, batch_index], 0)

            population_waves_interact = self.constraints(population_waves_interact, **kwargs)

            for h in range(n_populations):
                neuron_waves_interact[neurons_per_population*h : neurons_per_population*(h+1), :, batch_index] = \
                                     (population_waves_interact[h, :, batch_index]).repeat(neurons_per_population, 1)

            self.data[..., batch_index] = torch.poisson(neuron_waves_interact[..., batch_index]*dt)

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

    def get_random_gaussian(self, T, dt, fr_range, mu_range, std_range, n_range, peak_spread='max', **kwargs):

        T = np.arange(0, int(T / dt))
        n_min, n_max = int(n_range[0] / dt), int(n_range[1] / dt)
        mu_min, mu_max = mu_range[0] / dt, mu_range[1] / dt
        std_min, std_max = std_range[0] / dt, std_range[1] / dt

        def gaussian_pdf(x, mu, std):
            pdf = 1 / (np.sqrt(np.pi) * std) * np.exp(-0.5 * ((x - mu) / std) ** 2)
            return pdf

        if peak_spread == 'random':
            n_samples = int(np.random.rand(1) * (n_max - n_min) + n_min)
            trace = np.sum(gaussian_pdf(T[:, None], np.random.rand(1, n_samples) * (mu_max - mu_min) + mu_min,
                                        np.random.rand(1, n_samples) * (std_max - std_min) + std_min), 1)
        if peak_spread == 'max':
            fr = np.random.rand(1) * (n_range[1] - n_range[0]) + n_range[0]
            temp = np.random.poisson(fr, len(T))
            n_samples = np.sum(temp)
            i = 0
            while n_samples == 0:
                i += 1
                temp = np.random.poisson(fr, len(T))
                n_samples = np.sum(temp)
                if i == 100:
                    raise ValueError('Increase n_range (number of gaussian peaks)')

            mu = np.zeros(n_samples)
            freqm = 0
            for i, freq in enumerate(temp):
                if freq == 0:
                    continue
                mu[freqm:freqm + freq] = i
                freqm += freq
            trace = np.sum(gaussian_pdf(T[:, None], mu,
                                        np.random.rand(1, n_samples) * (std_max - std_min) + std_min), 1)

        max_fr = np.random.rand(1) * (fr_range[1] - fr_range[0]) + fr_range[0]

        return torch.tensor(trace / max(trace) * max_fr)

    def get_random_sine(self, T, dt, amplitude_range, frequency_range, phase_range):

        T = torch.linspace(start=0, end=2 * torch.pi, steps=int(T / dt))
        amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        frequency = torch.rand(1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]

        return amplitude * torch.sin(frequency * T - phase) + amplitude

    def create_random_connections(self, n_populations, fraction_exc_inh=0.5, max_correlation=0.9, min_correlation=0.6,
                                  sparcity=0.2, show_connection=False):
        N_E = int(fraction_exc_inh * n_populations)
        N_I = n_populations - N_E

        # Weight matrix
        U = np.random.normal(0.05, 0.02, [n_populations, n_populations])  # background activity
        U[U < 0] = 0
        U[N_E:n_populations, :] = - U[N_E:n_populations, :]

        for n in range(n_populations):
            ei = np.random.randint(low=0, high=N_E, size=(np.random.randint(low=N_E, high=N_E + 1),))
            ii = np.random.randint(low=0, high=N_I, size=(np.random.randint(low=N_I, high=N_I + 1),)) + N_E
            for j in range(len(ei)):
                U[ei[j], n] = np.random.rand(1) * (max_correlation - min_correlation) + min_correlation
            for j in range(len(ii)):
                U[ii[j], n] = -(np.random.rand(1) * (max_correlation - min_correlation) + min_correlation)

        if np.sum(U == 0)/n_populations**2 < sparcity:
            U.ravel()[np.random.permutation(n_populations ** 2)[:int(sparcity * n_populations ** 2)]] = 0

        U -= 0.8 * np.diag(np.diag(U))

        if show_connection:
            sns.heatmap(U.T)
            plt.show()

        return torch.tensor(U.T)

    def constraints(self, population_waves_interact, **kwargs):
        if torch.min(population_waves_interact) < 2 * kwargs['upper_bound_fr'] or torch.max(
                population_waves_interact) > 2 * kwargs['upper_bound_fr']:
            if torch.min(population_waves_interact) < 2 * kwargs['upper_bound_fr']:
                print(
                    'NOTE: Inhibitory connections are large, firing rate after interaction have reached >-{:.2f} Hz, but are now bounded'.format(
                        2 * kwargs['upper_bound_fr']))
            elif torch.max(population_waves_interact) > 2 * kwargs['upper_bound_fr']:
                print(
                    'NOTE: Excitatory connections are large, firing rate after interaction have reached <{:.2f} Hz, but are now bounded'.format(
                        2 * kwargs['upper_bound_fr']))
            else:
                print(
                    'NOTE: Excitatory * inhibitory connections are large, firing rate after interaction have reached <$\mid{:.2f}\mid$ Hz, but are now bounded'.format(
                        2 * kwargs['upper_bound_fr']))
            print('Hint: lower temporal connectivity (increase corr) or increase sparcity in the temporal connectivity')

        population_waves_interact[population_waves_interact < kwargs['lower_bound_fr']] = kwargs['lower_bound_fr']
        population_waves_interact[population_waves_interact > kwargs['upper_bound_fr']] = kwargs['upper_bound_fr']
        return population_waves_interact

    def plot_stats(self, batch=0, axes=None):
        if axes is None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.firing_rates[torch.argsort(self.firing_rates)], '.')
        axes[0, 0].set_title('Mean firing rates (over all batches)')
        sns.heatmap(self.data[..., batch], ax=axes[0, 1], cbar=False)
        axes[0, 1].set_title('Final spikes')

        for i, (wave_O, wave_I) in enumerate(zip(self.population_waves_original[..., batch, ], self.population_waves_interact[..., batch])):
            axes[1, 0].plot(wave_O[:self.time_steps_per_batch], label=str(i))
            axes[1, 1].plot(wave_I, label=str(i))

        axes[1, 0].set_title('Original population waves')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_xticks([0, self.time_steps_per_batch])
        axes[1, 0].set_xticklabels(['0', str(self.time_steps_per_batch)])

        axes[1, 1].set_title('Population waves after interaction')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_xticks([0, self.time_steps_per_batch])
        axes[1, 1].set_xticklabels([str(self.delay), str(self.time_steps_per_batch+self.delay)])

        return axes

if __name__ == '__main__':
    n_h = 6
    duration = 1
    dt = 1e-2
    x = PoissonTimeShiftedData(
        neurons_per_population=10,
        n_populations=n_h,
        n_batches=1,
        duration=duration, dt=dt,
        fr_mode='gaussian', delay=1, temporal_connections='random', corr=1, show_connection=True,
        fr_range=[50, 100], mu_range=[0, duration], std_range=[2 * dt, 5 * dt], n_range=[0.005, 0.05])


    axes = x.plot_stats()
    plt.show()

