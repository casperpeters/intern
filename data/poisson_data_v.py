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
            fr_mode='gaussian', delay=1, temporal_connections=None, corr=None, sparcity=0, compute_overlap=False,
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
                kwargs['fr_range'] = [50, 100]
            if 'mu_range' not in kwargs:
                kwargs['mu_range'] = [0, duration]  # in seconds
            if 'std_range' not in kwargs:
                kwargs['std_range'] = [2 * dt, 5 * dt]  # in seconds
            if 'n_range' not in kwargs:
                kwargs['n_range'] = [0.01, 0.05]  # average number of different gaussians per bin
            if 'lower_bound_fr' not in kwargs:
                kwargs['lower_bound_fr'] = 0
            if 'upper_bound_fr' not in kwargs:
                kwargs['upper_bound_fr'] = 2 * kwargs['fr_range'][1]
        kwargs['number of runs constraint'] = 0
        if corr == None:
            corr = n_populations ** 2

        # if no temporal connections are given, take half inhibitory and half exciting populations
        if temporal_connections == 'deterministic':
            if n_populations % 2 == 0: a=0
            else: a=1
            temporal_connections = torch.cat(
                (torch.ones(n_populations//2+a, n_populations),
                 torch.full(size=(n_populations//2, n_populations), fill_value=-1)), dim=0) / corr
            temporal_connections -= torch.diag(torch.diag(temporal_connections))

        if temporal_connections is None or temporal_connections == 'random':
            temporal_connections = self.create_random_connections(n_populations=n_populations, sparcity=sparcity) / corr

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
            for h in range(n_populations):
                # get a random sine wave or gaussian as mother train firing rate
                if fr_mode == 'sine':
                    population_waves_original[h, :, batch_index] = self.get_random_sine(T=duration+delay*dt, dt=dt, **kwargs)
                elif fr_mode == 'gaussian':
                    population_waves_original[h, :, batch_index] = self.get_random_gaussian(T=duration+delay*dt, dt=dt, **kwargs) + kwargs['lower_bound_fr']

            # compute interactions of all populations on their resulting firing rate
            population_waves_interact_temp = population_waves_original[..., batch_index].detach().clone()
            for t in range(delay, time_steps_per_batch+delay):
                for h in range(n_populations):
                    population_waves_interact_temp[h, t] += torch.sum(temporal_connections[:, h][None, :] * \
                                                                      population_waves_interact_temp[:, t-delay]) / n_populations
                # constrain to only positive values, upper limit and remove nan
                population_waves_interact_temp[:, t] = self.constraints(population_waves_interact_temp[:, t], **kwargs)

            # cut first temporal part (not every thing has temporal connections) and apply constraints
            population_waves_interact[..., batch_index] = population_waves_interact_temp[:, delay:]

            for h in range(n_populations):
                neuron_waves_interact[neurons_per_population*h : neurons_per_population*(h+1), :, batch_index] = \
                                     (population_waves_interact[h, :, batch_index]).repeat(neurons_per_population, 1)

            self.data[..., batch_index] = torch.poisson(neuron_waves_interact[..., batch_index]*dt)

        if compute_overlap: # compute only of the first batch
            self.overlap_fr = torch.zeros(n_populations)
            for h in range(n_populations):
                self.overlap_fr[h] = self.compute_overlap_fr(population_waves_original[h, delay:, 0],
                                                             population_waves_interact[h, :, 0],
                                                             n_bins=4 * time_steps_per_batch)
        else:
            if time_steps_per_batch > 100:
                T = 100
            else:
                T = time_steps_per_batch
            self.overlap_fr = torch.zeros(n_populations)
            for h in range(n_populations):
                self.overlap_fr[h] = self.compute_overlap_fr(population_waves_original[h, delay:T+delay, 0],
                                                             population_waves_interact[h, :T, 0],
                                                             n_bins=4 * T)

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

        T = np.arange(0, int(np.ceil(T / dt)))
        n_min, n_max = int(n_range[0] / dt), int(n_range[1] / dt)
        mu_min, mu_max = mu_range[0] / dt, mu_range[1] / dt
        std_min, std_max = std_range[0] / dt, std_range[1] / dt

        def gaussian_pdf(x, mu, std):
            pdf = 1 / (np.sqrt(np.pi) * std) * np.exp(-0.5 * ((x - mu) / std) ** 2)
            return pdf

        def check_length_sequence(array, th=2):
            max_lenght_sequence = 100
            while max_lenght_sequence > th:
                array_ = array > 0
                bounded = np.hstack(([0], array_, [0]))  # make sure all runs of ones are well-bounded
                difs = np.diff(bounded)  # get 1 at run starts and -1 at run ends
                run_starts, = np.where(difs > 0)
                run_ends, = np.where(difs < 0)
                length_sequence = run_ends - run_starts
                max_lenght_sequence = max(length_sequence)
                for i in range(len(length_sequence)):
                    if length_sequence[i] <= th:
                        continue
                    else:
                        array_[run_starts[i] + th] = abs(array_[run_starts[i] + th] - 1) > 0

                array_ = 1.0 * array_
                array_[array_ == 1] = array[array_ == 1]
                array = array_.copy()
            return np.array(array).astype(int)

        if peak_spread == 'random':
            n_samples = int(np.random.rand(1) * (n_max - n_min) + n_min)
            trace = np.sum(gaussian_pdf(T[:, None], np.random.rand(1, n_samples) * (mu_max - mu_min) + mu_min,
                                        np.random.rand(1, n_samples) * (std_max - std_min) + std_min), 1)
        if peak_spread == 'max':
            fr = np.random.rand(1) * (n_range[1] - n_range[0]) + n_range[0]
            temp = np.random.poisson(fr, len(T))
            temp[temp>2] = 2
            temp = check_length_sequence(temp, th=3)

            n_samples = np.sum(temp)
            i = 0
            while n_samples == 0:
                i += 1
                temp = np.random.poisson(fr, len(T))
                temp[temp > 2] = 2
                temp = check_length_sequence(temp, th=2)
                n_samples = np.sum(temp)
                if i == 100:
                    raise ValueError('Increase n_range (number of gaussian peaks)')
                    break

            mu = np.zeros(n_samples)
            freqm = 0
            for i, freq in enumerate(temp):
                if freq == 0:
                    continue
                mu[freqm:freqm + freq] = i
                freqm += freq
            trace = np.sum(gaussian_pdf(T[:, None], mu,
                                        np.random.rand(1, n_samples) * (std_max - std_min) + std_min), 1)

            # peaks = np.where((trace[1:-1] > trace[0:-2]) * (trace[1:-1] > trace[2:]))[0] + 1
            # maxi = max(trace[peaks])
            # mini = min(trace[peaks])

        max_fr = np.random.rand(1) * (fr_range[1] - fr_range[0]) + fr_range[0]
        return torch.tensor(trace / max(trace) * max_fr)

    def get_random_sine(self, T, dt, amplitude_range, frequency_range, phase_range):

        T = torch.linspace(start=0, end=2 * torch.pi, steps=int(T / dt))
        amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
        frequency = torch.rand(1) * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]

        return amplitude * torch.sin(frequency * T - phase) + amplitude

    def create_random_connections(self, n_populations, fraction_exc_inh=0.5, max_correlation=0.9, min_correlation=0.6, sparcity=0, self_conn=False):
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
        if isinstance(self_conn, (float)):
            return torch.tensor(U - self_conn * np.diag(np.diag(U)))
        elif self_conn==True or self_conn==None:
            return torch.tensor(U - 0.8 * np.diag(np.diag(U)))
        elif self_conn==False:
            return -torch.tensor(U - 1 * np.diag(np.diag(U)))

    def constraints(self, population_waves_interact, **kwargs):

        # if torch.min(population_waves_interact) < -2 * kwargs['upper_bound_fr'] and kwargs['number of runs constraint']==0 or \
        #         torch.max(population_waves_interact) > 2 * kwargs['upper_bound_fr'] and kwargs['number of runs constraint']==0:
        #
        #     if torch.min(population_waves_interact) < -2 * kwargs['upper_bound_fr']:
        #         print(
        #             'NOTE: Inhibitory connections are large, firing rate after interaction have reached >-{:.2f} Hz, but are now bounded'.format(
        #                 2 * kwargs['upper_bound_fr']))
        #     elif torch.max(population_waves_interact) > 2 * kwargs['upper_bound_fr']:
        #         print(
        #             'NOTE: Excitatory connections are large, firing rate after interaction have reached <{:.2f} Hz, but are now bounded'.format(
        #                 2 * kwargs['upper_bound_fr']))
        #     else:
        #         print(
        #             'NOTE: Excitatory * inhibitory connections are large, firing rate after interaction have reached <$\mid{:.2f}\mid$ Hz, but are now bounded'.format(
        #                 2 * kwargs['upper_bound_fr']))
        #     print('Hint: lower temporal connectivity (increase corr) or increase sparcity in the temporal connectivity')
        #     kwargs['number of runs constraint'] = 1
        population_waves_interact[population_waves_interact < kwargs['lower_bound_fr']] = kwargs['lower_bound_fr']
        population_waves_interact[population_waves_interact > kwargs['upper_bound_fr']] = kwargs['upper_bound_fr']
        return population_waves_interact

    def compute_overlap_fr(self, arr1, arr2, n_bins):
        # Determine the range over which the integration will occur
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)

        min_value = np.min((arr1.min(), arr2.min()))
        max_value = np.min((arr1.max(), arr2.max()))
        # Determine the bin width
        bin_width = (max_value - min_value) / n_bins
        # For each bin, find min frequency
        lower_bound = min_value  # Lower bound of the first bin is the min_value of both arrays
        min_arr = np.empty(n_bins)  # Array that will collect the min frequency in each bin

        for b in range(n_bins):
            higher_bound = lower_bound + bin_width  # Set the higher bound for the bin
            # Determine the share of samples in the interval
            freq_arr1 = np.ma.masked_where((arr1 < lower_bound) | (arr1 >= higher_bound), arr1).count() / len(arr1)
            freq_arr2 = np.ma.masked_where((arr2 < lower_bound) | (arr2 >= higher_bound), arr2).count() / len(arr2)

            # Conserve the lower frequency
            min_arr[b] = np.min((freq_arr1, freq_arr2))

            lower_bound = higher_bound  # To move to the next range
        return torch.tensor(min_arr.sum())

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
        for i, (wave_O, wave_I) in enumerate(zip(self.population_waves_original[..., batch], self.population_waves_interact[..., batch])):
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

        axes[1, 2].bar(np.arange(self.overlap_fr.shape[0]), np.sort(np.array(self.overlap_fr))[::-1])
        axes[1, 2].set_title('Overlap of pop. waves before and after interaction ')
        axes[1, 2].set_xlabel('Hidden population index')
        axes[1, 2].set_ylabel('Normalized overlap')
        axes[1, 2].set_ylim([0, 1])

        return axes

if __name__ == '__main__':
    n_h = 3
    duration = 1
    dt = 1e-2
    corr=0.5
    s = PoissonTimeShiftedData(
        neurons_per_population=20,
        n_populations=n_h,
        n_batches=1,
        duration=duration, dt=dt,
        fr_mode='gaussian', delay=1, temporal_connections='random', corr=corr, show_connection=False,
        compute_overlap=False,
        fr_range=[40, 100], mu_range=[0, duration], std_range=[1 * dt, 2 * dt], n_range=[0.048, 0.05])

    axes = s.plot_stats()
    plt.show()

