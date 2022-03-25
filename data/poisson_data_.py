import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PoissonTimeShiftedData(object):
    def __init__(
            self,
            neurons_per_population=10,
            n_populations=20,
            n_batches=200,
            duration=0.1, dt=1e-2,
            fr_mode='sine', delay=1, temporal_connections=None, corr=None, show_connection=True,
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
                kwargs['fr_range'] = [20, 21]
            if 'mu_range' not in kwargs:
                kwargs['mu_range'] = [0, duration] #in seconds
            if 'std_range' not in kwargs:
                kwargs['std_range'] = [2*dt, 10*dt] #in seconds
            if 'n_range' not in kwargs:
                kwargs['n_range'] = [0.5, 1.5] # average number of different gaussians per bin

        if corr==None:
            corr = n_populations**2

        # if no temporal connections are given, take half inhibitory and half exciting populations
        if temporal_connections is None:
            temporal_connections = torch.cat(
                (torch.ones(n_populations, n_populations // 2),
                 torch.full(size=(n_populations, n_populations // 2), fill_value=-1)),
                dim=1
            ) / corr

        if temporal_connections == 'random':
            temporal_connections = self.create_random_connections(n_populations=n_populations, show_connection=show_connection) / corr

        neuron_indexes = torch.arange(end=neurons_per_population * n_populations).view(n_populations,
                                                                                       neurons_per_population)
        # initialize empty tensors
        time_steps_per_batch = int(duration/dt)

        self.data = torch.empty(
            neurons_per_population * n_populations,
            time_steps_per_batch,
            n_batches,
            dtype=torch.float
        )

        self.deleted_spikes = torch.empty_like(self.data)
        self.added_spikes = torch.empty_like(self.data)
        self.population_waves = torch.empty(n_populations, time_steps_per_batch, n_batches)
        self.mother_trains = torch.empty(n_populations, time_steps_per_batch, n_batches)

        # evenly spaced firing rate per population
        if fr_mode=='gaussian':
            fr_per_pop = np.zeros([2, n_populations])
            fr_per_pop[0, :] = np.random.rand(1, n_populations) * (kwargs['fr_range'][1]/2 - kwargs['fr_range'][0]) + kwargs['fr_range'][0]
            fr_per_pop[1, :] = np.random.rand(1, n_populations) * (kwargs['fr_range'][1]/2 - kwargs['fr_range'][0]/4) + kwargs['fr_range'][0]/4

        # loop over batches
        for batch_index in range(n_batches):

            # get all mother trains by looping over populations
            for population_index in range(n_populations):
                # get a random sine wave as mother train firing rate
                if fr_mode == 'sine':
                    population_wave = self.get_random_sine(T=duration, dt=dt, **kwargs)
                elif fr_mode == 'gaussian':
                    kwargs['fr_range'] = [fr_per_pop[0, population_index], fr_per_pop[0, population_index] + fr_per_pop[1, population_index]]
                    population_wave = self.get_random_gaussian(T=duration, dt=dt, **kwargs)

                # poisson draw mother train
                mother_train = torch.poisson(population_wave * dt)

                # assign population wave and mother train to tensors
                self.population_waves[population_index, :, batch_index] = population_wave
                self.mother_trains[population_index, :, batch_index] = mother_train

            batch = torch.zeros(neurons_per_population * n_populations, time_steps_per_batch)

            # get spikes
            for population_index in range(n_populations):
                connections = temporal_connections[population_index, :]
                population_waves = self.population_waves[..., batch_index]

                # delete spikes based on inhibitory connections
                inhibitory_connections = connections < 0
                deletion_probabilities = -1 * torch.sum(
                    input=connections[inhibitory_connections].unsqueeze(1) * population_waves[inhibitory_connections, :] * dt,
                    dim=0
                ).T
                delete_spikes = (torch.tile(
                    input=torch.roll(
                        input=deletion_probabilities,
                        shifts=delay,
                        dims=0
                    ),
                    dims=(neurons_per_population, 1)
                ) > torch.rand(neurons_per_population, time_steps_per_batch)).type(torch.float)

                # add spikes based on exciting connections
                exciting_connection = connections > 0
                addition_probabilities = torch.sum(
                    input=connections[exciting_connection].unsqueeze(1) * population_waves[exciting_connection, :] * dt,
                    dim=0
                ).T

                add_spikes = torch.poisson(
                    torch.tile(
                        input=torch.roll(
                            input=addition_probabilities,
                            shifts=delay,
                            dims=0
                        ),
                        dims=(neurons_per_population, 1)
                    )
                ).type(torch.float)

                add_spikes[add_spikes > 1] = 1

                spikes = torch.tile(
                    input=self.mother_trains[population_index, :, batch_index],
                    dims=(neurons_per_population, 1)
                ) - delete_spikes + add_spikes

                batch[neuron_indexes[population_index], :] = spikes[torch.argsort(torch.mean(spikes, dim=1)), :]

                self.deleted_spikes[neuron_indexes[population_index], :, batch_index] = delete_spikes
                self.added_spikes[neuron_indexes[population_index], :, batch_index] = add_spikes

            self.data[..., batch_index] = batch

        # make sure there are
        self.data[self.data < 0] = 0
        self.data[self.data > 1] = 1
        self.firing_rates = torch.mean(self.data, (1, 2))

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
            mu = np.zeros(n_samples)
            freqm = 0
            for i, freq in enumerate(temp):
                if freq == 0:
                    continue
                mu[freqm:freqm + freq] = i
                freqm += freq
            trace = np.sum(gaussian_pdf(T[:, None], mu,
                                        np.random.rand(1, n_samples) * (std_max - std_min) + std_min), 1)

        # shift to 0
        max_fr = np.random.rand(1) * (fr_range[1] - fr_range[0])  # + fr_range[0]
        # normalize and shift to right firing range

        return torch.tensor(trace / max(trace) * max_fr + fr_range[0])

    def get_random_sine(self, T, dt, amplitude_range, frequency_range, phase_range):

        T = torch.linspace(start=0, end=2 * torch.pi, steps=int(T/dt))
        amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        frequency = torch.rand(1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]

        return amplitude * torch.sin(frequency * T - phase) + amplitude

    def plot_stats(self, batch=0, axes=None):
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        sns.heatmap(self.data[..., batch], ax=axes[0, 0], cbar=False)
        axes[0, 0].set_title('Final spikes')
        sns.heatmap(self.deleted_spikes[..., batch], ax=axes[0, 1], cbar=False)
        axes[0, 1].set_title('Deleted spikes')
        sns.heatmap(self.added_spikes[..., batch], ax=axes[0, 2], cbar=False)
        axes[0, 2].set_title('Added spikes')
        colors=['green', 'red', 'blue', 'orange', 'black', 'green', 'red', 'blue', 'orange', 'black']
        for batch in range(100):
            for i, wave in enumerate(self.population_waves[..., batch]):
                axes[1, 0].plot(wave, color = colors[i])
        axes[1, 0].set_title('Population waves')

        sns.heatmap(self.mother_trains[..., batch], ax=axes[1, 1])
        axes[1, 1].set_title('Mother trains')
        axes[1, 2].plot(self.firing_rates[torch.argsort(self.firing_rates)], '.')
        axes[1, 2].set_title('Mean firing rates (over all batches)')
        return axes

    def create_random_connections(self, n_populations, fraction_exc_inh=0.5, max_correlation=0.9, min_correlation=0.6, n_excitatory_input = 1, n_inhibitory_input=1, show_connection=False):
        N_E = int(fraction_exc_inh * n_populations)
        N_I = n_populations - N_E

        min_n_inhibitory_input = int(0.9 * N_E)
        min_n_excitatory_input = int(0.9 * N_I)
        # Weight matrix
        W = np.random.normal(0.05, 0.02, [n_populations, n_populations])  # background activity
        W[W < 0] = 0
        W[N_E:n_populations, :] = - W[N_E:n_populations, :]

        for n in range(n_populations):
            ei = np.random.randint(low=0, high=N_E, size=(np.random.randint(low=min_n_excitatory_input, high=N_E+1),))
            ii = np.random.randint(low=0, high=N_I, size=(np.random.randint(low=min_n_inhibitory_input, high=N_I+1),)) + N_E
            for j in range(len(ei)):
                W[ei[j], n] = np.random.rand(1) * (max_correlation - min_correlation) + min_correlation
            for j in range(len(ii)):
                W[ii[j], n] = -(np.random.rand(1) * (max_correlation - min_correlation) + min_correlation)

        self.W = torch.tensor(W.T)
        self.W = self.W - 0.5 * torch.diag(torch.diag(self.W))

        if show_connection:
            sns.heatmap(self.W)
            plt.show()

        return self.W

if __name__ == '__main__':

    n_populations = 5
    neurons_per_population = 10
    x = PoissonTimeShiftedData(
        n_batches=100,
        n_populations=n_populations,
        neurons_per_population=neurons_per_population,
        fr_mode='gaussian',
        duration=1, dt=1e-2, # duration in seconds
        temporal_connections='random'
    )

    print(x.data.mean())
    axes = x.plot_stats()
    plt.show()

    # PMT = x.mother_trains
    # modulated_PT = x.data_poisson
    #
    # _, ax = plt.subplots(1, 2)
    # for i in range(n_populations):
    #     ax[0].plot(PMT[i, :, 0])
    #
    # for i in range(n_populations):
    #     ax[1].plot(modulated_PT[int(neurons_per_population * i), :, 0])
    # plt.show()

