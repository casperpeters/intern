import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class PoissonTimeShiftedData(object):
    def __init__(
            self,
            neurons_per_population=10,
            n_populations=20,
            n_batches=200,
            time_steps_per_batch=100,
            delay=1,
            population_wave_form='sin',  # ('sin', 'gaussian')
            temporal_connections=None,
            amplitude_range=None,
            frequency_range=None,
            n_peaks_range=None,
            bandwidth=None,
    ):

        """
        """

        # if no temporal connections are given, take half inhibitory and half exciting populations
        if population_wave_form == 'sin':
            if frequency_range is None:
                frequency_range = [2, 5]
            if amplitude_range is None:
                amplitude_range = [.05, .5]
        elif population_wave_form == 'gaussian':
            if n_peaks_range is None:
                n_peaks_range = [2, 10]
            if bandwidth is None:
                bandwidth = 5
            if amplitude_range is None:
                amplitude_range = [.1, .6]

        if temporal_connections is None:
            temporal_connections = torch.cat(
                tensors=(torch.ones(n_populations, n_populations // 2),
                         torch.full(size=(n_populations, n_populations // 2), fill_value=-1)),
                dim=1
            ) / (n_populations * 2)

        self.time_steps_per_batch = time_steps_per_batch
        self.t = torch.linspace(start=0, end=2 * torch.pi, steps=time_steps_per_batch)
        neuron_indexes = torch.arange(end=neurons_per_population * n_populations).view(n_populations,
                                                                                       neurons_per_population)

        # initialize empty tensors
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

        # loop over batches
        for batch_index in range(n_batches):

            # get all mother trains by looping over populations
            for population_index in range(n_populations):
                if population_wave_form == 'sin':
                    # get a random sine wave as mother train firing rate
                    population_wave = self.get_random_sine(
                        amplitude_range=amplitude_range,
                        frequency_range=frequency_range,
                        phase_range=[0, torch.pi]
                    )
                elif population_wave_form == 'gaussian':
                    population_wave = self.get_random_gaussians(
                        n_peaks_range=n_peaks_range,
                        bandwidth=bandwidth,
                        amplitude_range=amplitude_range
                    )

                # poisson draw mother train
                mother_train = torch.poisson(population_wave)

                # assign population wave and mother train to tensors
                self.population_waves[population_index, :, batch_index] = population_wave
                self.mother_trains[population_index, :, batch_index] = mother_train

            # empty tensor for this batch
            batch = torch.empty(neurons_per_population * n_populations, time_steps_per_batch)

            # get spikes by looping over populations
            for population_index in range(n_populations):
                # connections to current population
                connections = temporal_connections[population_index, :]
                population_waves = self.population_waves[..., batch_index]

                # delete spikes based on inhibitory connections
                inhibitory_connections = connections < 0
                deletion_probabilities = -1 * torch.sum(
                    input=connections[inhibitory_connections].unsqueeze(1) *
                          population_waves[inhibitory_connections, :],
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
                    input=connections[exciting_connection].unsqueeze(1) * population_waves[exciting_connection, :],
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

    def get_random_sine(
            self,
            amplitude_range,
            frequency_range,
            phase_range,
    ):
        """
        """
        amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        frequency = torch.rand(1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]

        return amplitude * torch.sin(frequency * self.t - phase) + amplitude

    def get_random_gaussians(
            self,
            n_peaks_range,
            bandwidth,
            amplitude_range
    ):
        """
        """
        amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        n_peaks = np.random.randint(low=n_peaks_range[0], high=n_peaks_range[1])
        peak_locations = np.random.randint(low=0, high=self.time_steps_per_batch, size=n_peaks)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(peak_locations[:, None])
        gausfit = kde.score_samples(np.arange(start=0, stop=self.time_steps_per_batch)[:, None])
        wave = torch.tensor(np.exp(gausfit))
        return wave / torch.max(wave) * amplitude

    def plot_stats(self, batch=0, axes=None):
        """
        plots the added, deleted and final spikes of the class of given batch. Also plots the population waves, mother
        trains and the mean firing rates (over all batches).

        Parameters
        ----------
        batch : int
            to batch number of the data to plot, must be < self.n_batches
        axes : matplotlib.pyplot.axes, optional
            when given, uses these axes to perform the plot on, axes.shape() must be (2, 3)
        """
        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        elif axes.shape != (2, 3):
            raise ValueError('axes must be None or of shape (2, 3)')
        axes[0, 0].imshow(self.data[..., batch], cmap=plt.get_cmap('binary'), aspect='auto')
        axes[0, 0].set_title('Final spikes')
        axes[0, 1].imshow(self.deleted_spikes[..., batch], cmap=plt.get_cmap('binary'), aspect='auto')
        axes[0, 1].set_title('Deleted spikes')
        axes[0, 2].imshow(self.added_spikes[..., batch], cmap=plt.get_cmap('binary'), aspect='auto')
        axes[0, 2].set_title('Added spikes')
        for wave in self.population_waves[..., batch]:
            axes[1, 0].plot(wave)
        axes[1, 0].set_title('Population waves')
        axes[1, 1].imshow(self.mother_trains[..., batch], cmap=plt.get_cmap('binary'), aspect='auto')
        axes[1, 1].set_title('Mother trains')
        axes[1, 2].plot(self.firing_rates[torch.argsort(self.firing_rates)], '.')
        axes[1, 2].set_title('Mean firing rates (over all batches)')
        return axes

    def add_noise(self, sigma=.001, range=.5, delete_spikes=True):
        sigma = torch.linspace(sigma-sigma*range, sigma+sigma*range, self.data.shape[0]).unsqueeze(1).unsqueeze(2)
        add = (torch.rand(self.data.shape) < sigma.tile(1, self.data.shape[1], self.data.shape[2])).type(torch.float)

        self.data += add
        if delete_spikes:
            delete = (torch.rand(self.data.shape) < sigma).type(torch.float)
            self.deleted_spikes += delete
            self.data -= delete
        self.added_spikes += add
        self.data[self.data < 0] = 0
        self.data[self.data > 1] = 1
        self.firing_rates = torch.mean(self.data, (1, 2))
        return


if __name__ == '__main__':
    temporal_conn = torch.tensor([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])

    x = PoissonTimeShiftedData(
        neurons_per_population=7,
        n_populations=3,
        n_batches=500,
        time_steps_per_batch=30,
        temporal_connections=temporal_conn,
        amplitude_range=[.05, .3],
        population_wave_form='gaussian',
        bandwidth=1,
    )
    x.add_noise()
    axes = x.plot_stats()
    plt.show()
