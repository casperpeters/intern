import torch
import numpy as np
from scipy import signal
import random


def create_complex_artificial_data(n_neurons=900,
                                   t_max=2500,
                                   n_populations=9,
                                   mean_firing_rate=0.1,
                                   population_correlations=None,
                                   neuron_population_correlation=None,
                                   time_shifts=None,
                                   permute=True
                                   ):
    """
    Create surrogate data of 9 neuron populations with different correlations:
    population 1: no correlation, random spiking
    population 2 - 9: high inter-neuron correlation (0.9)
    population 2: no correlation to other populations
    population 3 - 9: these populations are spatially correlated (.5, .5, .8, .8, .8, .8, .8)
    population 5 - 9: have additional time shifted spikes, which makes them temporally correlated

    These correlations can be adjusted:
        - population_correlations: defines the correlations between populations
        - neuron_population_correlation: defines inter-neuron correlation for every population
        - time-shifts: gives a population a certain lag
        - permute: if true, shuffle neuron indexes

    returns:
        - data: spiking data of all neurons
        - coordinates: brain-space coordinates of these neurons
        - population_idx: np.array of np.arrays of population neuron indexes
    """

    if time_shifts is None:
        time_shifts = [0, 0, 0, 0, 0, -2, -1, 1, 2]
    if neuron_population_correlation is None:
        neuron_population_correlation = [0, .7, .7, .7, .7, .7, .7, .7, .7]
    if population_correlations is None:
        population_correlations = [0, 0, .5, .5, .8, .8, .8, .8, .8]

    n_neurons_pop = int(n_neurons / n_populations)

    spikes = np.zeros((n_neurons, t_max))

    # population 1: uncorrelated
    spikes[:n_neurons_pop, :] = np.random.poisson(lam=mean_firing_rate, size=(n_neurons_pop, t_max))

    # population 2 - 9: correlated
    poisson_grandmother_train = np.random.poisson(lam=mean_firing_rate, size=(1, t_max))
    poisson_mother_trains = np.zeros((n_populations, t_max))
    for i in range(n_populations):
        # population dynamics
        delete_spikes = population_correlations[i] * np.ones((1, t_max)) <= np.random.uniform(0, 1, size=(1, t_max))
        noise = np.random.poisson((1 - population_correlations[i]) * mean_firing_rate,
                                  (1, t_max))  # add noise according to corr
        poisson_mother_trains[i] = poisson_grandmother_train - (
                delete_spikes * poisson_grandmother_train) + noise  # delete spikes, add noise
        poisson_mother_trains[i] = np.roll(poisson_mother_trains[i], time_shifts[i])  # add time shifts

        # neuron dynamics
        delete_spikes = neuron_population_correlation[i] * np.ones((n_neurons_pop, t_max)) <= \
                        np.random.uniform(0, 1, size=(n_neurons_pop, t_max))
        noise = np.random.poisson((1 - neuron_population_correlation[i]) * mean_firing_rate, (n_neurons_pop, t_max))
        population_spikes = np.tile(poisson_mother_trains[i], (n_neurons_pop, 1))
        population_spikes = population_spikes - (delete_spikes * population_spikes) + noise

        idx = np.arange(n_neurons_pop) + i * n_neurons_pop
        spikes[idx, :] = population_spikes

    spikes[spikes < 0] = 0
    spikes[spikes > 1] = 1

    # create 2 dimensional brain-space locations for all neuron populations
    mean_locations_pop = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    neuron_coordinates = np.zeros((n_neurons, 2))
    for i in range(n_populations):
        neuron_coordinates[n_neurons_pop * i:n_neurons_pop * (i + 1), :] = 0.25 * np.random.randn(n_neurons_pop, 2) + \
                                                                           mean_locations_pop[i]

    # randomly permute neurons
    populations_idx = np.arange(n_neurons, dtype=int)
    if permute:
        idx = np.random.permutation(n_neurons)
        neuron_coordinates = neuron_coordinates[idx, :]
        populations_idx = populations_idx[np.argsort(idx)]
        spikes = spikes[idx, :]
    populations_idx = populations_idx.reshape(n_populations, n_neurons_pop)

    return spikes, neuron_coordinates, populations_idx


def cai():
    n_neurons = 900
    t_max = 2500
    n_populations = 9
    permute = True
    time_shifts = [0, 0, 0, 0, 0, 0, 1, 2, 3]
    neuron_population_correlation = [0, .8, .8, .8, .8, .8, .8, .8, .8]
    population_correlations = [0, .6, .6, .6, .6, .6, .9, .9, .9]
    mean_firing_rate = 0.1

    n_neurons_pop = int(n_neurons / n_populations)

    spikes = np.zeros((n_neurons, t_max))

    # population 1 - 3: uncorrelated
    spikes[:n_neurons_pop * 3, :] = np.random.poisson(lam=mean_firing_rate, size=(n_neurons_pop * 3, t_max))

    # population 4 - 6: correlated
    poisson_grandmother_train1 = np.random.poisson(lam=mean_firing_rate, size=(1, t_max))
    poisson_grandmother_train2 = np.random.poisson(lam=mean_firing_rate, size=(1, t_max))

    poisson_mother_trains = np.zeros((n_populations, t_max))
    for i in range(3, 9):

        delete_spikes = population_correlations[i] * np.ones((1, t_max)) <= np.random.uniform(0, 1, size=(1, t_max))
        noise = np.random.poisson((1 - population_correlations[i]) * mean_firing_rate, (1, t_max))
        if i <= 5:
            poisson_mother_trains[i] = poisson_grandmother_train1 - (delete_spikes * poisson_grandmother_train1) + noise
        else:
            poisson_mother_trains[i] = poisson_grandmother_train2 - (delete_spikes * poisson_grandmother_train2) + noise
        poisson_mother_trains[i] = np.roll(poisson_mother_trains[i], time_shifts[i])  # add time shifts

        # neuron dynamics
        delete_spikes = neuron_population_correlation[i] * np.ones((n_neurons_pop, t_max)) <= \
                        np.random.uniform(0, 1, size=(n_neurons_pop, t_max))
        noise = np.random.poisson((1 - neuron_population_correlation[i]) * mean_firing_rate, (n_neurons_pop, t_max))
        population_spikes = np.tile(poisson_mother_trains[i], (n_neurons_pop, 1))
        population_spikes = population_spikes - (delete_spikes * population_spikes) + noise

        idx = np.arange(n_neurons_pop) + i * n_neurons_pop
        spikes[idx, :] = population_spikes

    # make binary
    spikes[spikes < 0] = 0
    spikes[spikes > 1] = 1

    # create 2 dimensional brain-space locations for all neuron populations
    mean_locations_pop = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    neuron_coordinates = np.zeros((n_neurons, 2))
    for i in range(n_populations):
        neuron_coordinates[n_neurons_pop * i:n_neurons_pop * (i + 1), :] = 0.25 * np.random.randn(n_neurons_pop, 2) + \
                                                                           mean_locations_pop[i]

    # randomly permute neurons
    populations_idx = np.arange(n_neurons, dtype=int)
    if permute:
        idx = np.random.permutation(n_neurons)
        neuron_coordinates = neuron_coordinates[idx, :]
        populations_idx = populations_idx[np.argsort(idx)]
        spikes = spikes[idx, :]
    populations_idx = populations_idx.reshape(n_populations, n_neurons_pop)
    return torch.tensor(spikes), torch.tensor(neuron_coordinates), populations_idx


def create_BB(N_V=16, T=32, n_samples=256, width_vec=[4, 5, 6, 7], velocity_vec=[1, 2], boundary=False, r=2):
    """ Generate 1 dimensional bouncing ball data with or without boundaries, with different ball widths and velocities"""

    data = np.zeros([N_V, T, n_samples])

    for i in range(n_samples):
        if boundary:
            v = random.sample(velocity_vec, 1)[0]
            dt = 1
            x = np.random.randint(r, N_V - r)
            trend = (2 * np.random.randint(0, 2) - 1)
            for t in range(T):
                if x + r > N_V - 1:
                    trend = -1
                elif x - r < 1:
                    trend = 1
                x += trend * v * dt

                data[x - r:x + r, t, i] = 1
        else:
            ff0 = np.zeros(N_V)
            ww = random.sample(width_vec, 1)[0]
            ff0[0:ww] = 1  # width

            vv = random.sample(velocity_vec, 1)[0]  # initial speed, vv>0 so always going right
            for t in range(T):
                ff0 = np.roll(ff0, vv)
                data[:, t, i] = ff0

    return torch.tensor(data, dtype=torch.float)


def generate_mock_data(N_V, T, sampling_frequency=4, average_firing_rate=0.0213, stimulus_period=480 / 4,
                       random_seed=0):
    """ Generate spike train data of N_V neurons for T timesteps with a given sampling frequency (Hz) and avarage firing rate.
        Stimulus period is taken from Misha's data.
        Returns spike trains and time array. """

    # set random seed
    torch.random.manual_seed(random_seed)

    # calculate stimulus frequency
    stimulus_frequency = 1 / stimulus_period

    # create time array
    time_arr = torch.linspace(0, T / sampling_frequency, T)

    # calculate firing probability with triangle wave
    firing_probability = torch.tensor(
        average_firing_rate * (signal.sawtooth(2 * np.pi * stimulus_frequency * time_arr, 0.5) + 1))

    # spike trains
    spikes = torch.rand(N_V, T) < firing_probability.expand(N_V, T)
    spikes = spikes.type(torch.DoubleTensor)

    return spikes, time_arr


def generate_data_fr_dist(N_V, T, spikes_true, true_sampling_rate=4, nr_of_discretized_steps=1000, randperm=True):
    """ This function creates mock data, but with the right firing rate distribution as given by spikes_true data. """

    # sort the spikes according to firing rate
    f_rate_order_true = torch.argsort(torch.sum(spikes_true, 1), descending=True)
    spikes_true = spikes_true[f_rate_order_true, :]

    # create empty data tensor
    data = torch.zeros(N_V, T)

    # loop over number of discretized steps
    for i in range(0, nr_of_discretized_steps):
        # an indexer
        idx = int(spikes_true.shape[0] / nr_of_discretized_steps)

        # calculate mean firing rate of spikes_true batch
        mean_firing_rate_discretized = torch.mean(spikes_true[i * idx:(i + 1) * idx, :])

        # use mean firing rate to generate mock data
        data[int(i * N_V / nr_of_discretized_steps):int((i + 1) * N_V / nr_of_discretized_steps), :], time_arr = \
            generate_mock_data(int(N_V / nr_of_discretized_steps), T,
                               average_firing_rate=mean_firing_rate_discretized * true_sampling_rate)

    # randomly permute data
    if randperm == True:
        data = data[torch.randperm(data.shape[0]), :]

    return data


def create_weights_binary_network(N=10, phigh=1.2, ilow=-4):
    # initialize NxN weight matrix
    Wa = np.random.uniform(high=phigh, size=(N // 2, N // 2))
    Wb = np.random.uniform(high=phigh, size=(N // 2, N // 2))
    for i in range(N // 2):
        Wa[i, i] = 0
        Wb[i, i] = 0

    # create large matrix W
    W = np.zeros(shape=(N, N))
    W[0:N // 2, 0:N // 2] = Wa
    W[N // 2:N, N // 2:N] = Wb

    # add negative connection between subgroups
    Wc = np.random.uniform(low=ilow, high=0, size=(N // 2, N // 2))
    Wd = np.random.uniform(low=ilow, high=0, size=(N // 2, N // 2))
    W[0:N // 2, N // 2:N] = Wc
    W[N // 2:N, 0:N // 2] = Wd

    return W

"""
# THIS PROGRAM DEMONSTRATES HODGKIN HUXLEY MODEL IN CURRENT CLAMP EXPERIMENTS AND SHOWS ACTION POTENTIAL PROPAGATION
# Time is in secs, voltage in mvs, conductances in m mho/mm^2, capacitance in uF/mm^2

# threshold value of current is 0.0223


import numpy as np
import matplotlib.pyplot as plt

g_K_max = .36  # max conductance of K channel
V_K = -77  # voltage of K channel
g_Na_max = 1.20  # max conductance of Na channel
V_Na = 50  # voltage of Na channel
g_l = 0.003  # conductance of combined gates
v_l = -54.387  # voltageof combined channel
cm = .01

dt = 0.01  # 0.01 ms
niter = 10000
t = np.array([i for i in range(niter)])
I_app = ImpCur * np.ones(niter)
V = -64.9964  # base voltage
m = 0.0530
h = 0.5960
n = 0.3177

#### to store the values
g_Na_hist = np.zeros(niter)
g_K_hist = np.zeros(niter)
V_hist = np.zeros(niter)
m_hist = np.zeros(niter)
h_hist = np.zeros(niter)
n_hist = np.zeros(niter)

for i in range(niter):
    g_Na = g_Na_max * (m ** 3) * h
    g_K = g_K_max * (n ** 4)
    g_total = g_Na + g_K + g_l
    V_inf = ((g_Na * V_Na + g_K * V_K + g_l * V_l) + I_app[i]) / g_total
    tau_v = cm / g_total
    V = V_inf + (V - V_inf) * np.exp(-dt / tau_v)
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4 * np.exp(-0.0556 * (V + 65))
    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)
    alpha_h = 0.07 * np.exp(-0.05 * (V + 65))
    beta_h = 1 / (1 + np.exp(-0.1 * (V + 35)))
    tau_m = 1 / (alpha_m + beta_m)
    tau_h = 1 / (alpha_h + beta_h)
    tau_n = 1 / (alpha_n + beta_n)
    m_inf = alpha_m * tau_m
    h_inf = alpha_h * tau_h
    n_inf = alpha_n * tau_n
    m = m_inf + (m - m_inf) * exp(-dt / tau_m)
    h = h_inf + (h - h_inf) * exp(-dt / tau_h)
    n = n_inf + (n - n_inf) * exp(-dt / tau_n)
    V_hist[i] = V
    m_hist[i] = m
    h_hist[i] = h
    n_hist[i] = n

plt.plot(t, V_hist)
plt.show()






"""




















