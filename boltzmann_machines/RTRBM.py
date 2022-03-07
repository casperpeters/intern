
"""
TODO
- Look at parallel tempering and augmented PT (particle is stuck in local minima
_ dReLU HU potential
- Centered dReLU HU potential
- PCD

"""

import torch
from tqdm import tqdm
from optim.lr_scheduler import get_lrs


class RTRBM(object):

    def __init__(self, data, N_H=10, device=None, init_biases=False, debug=False):

        """     __init__

                    Parameters
                    ----------
                   data : torch.tensor
                          Input/training data
                    N_H : int
                          Number of hidden units
                 device : str
                          Run code on CPU or GPU
           init_biases  : Boolean
                          Initialize biases such that sampling the visibles (p(v_i=1)) or hiddens (p(h_i=1))
                          on just the biases results in p(v_i=1) = <v_i> and p(h_i=1) = <h_i>

                    Returns
                    -------
                   self : Initialized parameters to describe the RTRBM
        """

        if device is None:
            self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        else:
            self.device = device

        self.dtype = torch.float
        self.V = data.float().to(self.device)
        self.debug = debug
        self.dim = torch.tensor(self.V.shape).shape[0]

        if self.dim == 2:
            self.N_V, self.T = self.V.shape
            self.num_samples = 1
        elif self.dim == 3:
            self.N_V, self.T, self.num_samples = self.V.shape
        else:
            raise ValueError("Data is not correctly defined: Use (N_V, T) or (N_V, T, num_samples) dimensions")

        self.N_H = N_H

        self.W = 0.01/self.N_V * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.U = 0.01/self.N_H * torch.randn(self.N_H, self.N_H, dtype=self.dtype, device=self.device)
        self.b_H = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(1, self.N_V, dtype=self.dtype, device=self.device)
        self.b_init = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)

        if init_biases:
            self.b_V = -torch.log(1 / torch.mean(data.reshape(self.N_V, self.T * self.num_samples), 1) - 1)[None, :].to(self.device)
            self.b_V[torch.isnan(self.b_V)] = 0.01 * torch.randn(1).to(self.device)
            self.b_V[self.b_V > 0.1] = 0.1 * torch.rand(1).to(self.device)
            self.b_V[self.b_V < 0.1] = -0.1 * torch.rand(1).to(self.device)
            mu_H = torch.mean(self.visible_to_expected_hidden(data.reshape(self.N_V, self.T * self.num_samples)), 1).to(self.device)
            self.b_H = -torch.log(1 / mu_H - 1)[None, :].to(self.device)
            self.b_H[torch.isnan(self.b_H)] = 0.01 * torch.randn(1).to(self.device)
            self.b_H[self.b_H > 0.1] = 0.1 * torch.rand(1).to(self.device)
            self.b_H[self.b_H < 0.1] = -0.1 * torch.rand(1).to(self.device)

            self.b_V = self.b_V.to(self.device)
            self.b_H = self.b_H.to(self.device)

        self.params = [self.W, self.U, self.b_H, self.b_V, self.b_init]

        if self.debug:
            self.parameter_history = []
            self.parameter_history.append(self.params)

    def learn(self,
              n_epochs=1000,
              batchsize=128,
              CDk=10, PCD=False,
              min_lr=1e-3, max_lr=None, lr_mode=None,
              sp=None, x=2,
              mom=0.9,
              wc=0.0002,
              AF=torch.sigmoid,
              U_normalisation=False,
              disable_tqdm=False, **kwargs):

        """     Training of the RTRBM

                    Parameters
                    ----------
               n_epochs : int
                          Number of epochs for training
              batchsize : int
                          The number of batchsizes used for computing the mean gradient before updating the RTRBM
                    CDk : int
                          The number of Gibbs sampling steps used to compute the negative phase in CD-k
                   PCD  : int
                          Persistent contrastive divergence (not working, most likely a bug)
                     lr : float
                          Learning rate of gradient decent
                 lr_end : float
                          iff None -> no learning rate decay
                          Geometrically decay of learning to an end value lr_end at epoch n_epochs
            start_decay : float
                          Start decay at epoch number start_decay
                     sp : float
                          Sparse penalty on the visible to hidden weight
                      x : int
                          Polynomial proportionality of the sparse penalty relative to the weights
                   mom  : float
                          Momentum of gradient decent
                     wc : float
                          Weightcost of gradient decent
                     AF : function
                          The activation function.
        U_normalisation : Boolean
                          Normalize the U matrix each epoch
           disable_tqdm : Boolean
                          Progress bar on (False) or off (True)

                    Returns
                    -------
                    self
        """

        global vt_k
        if self.dim == 2:
            num_batches = 1
            batchsize = 1
        elif self.dim == 3:
            num_batches = self.num_samples // batchsize

        # get learning rate schedule
        if lr_mode is None:
            lrs = min_lr * torch.ones(n_epochs)
        else:
            lrs = get_lrs(mode=lr_mode, n_epochs=n_epochs, min_lr=min_lr, max_lr=max_lr, **kwargs)

        Dparams = self.initialize_grad_updates()

        self.errors = torch.zeros(n_epochs, 1)
        self.disable = disable_tqdm

        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0

            for batch in range(0, num_batches):

                self.dparams = self.initialize_grad_updates()

                for i in range(0, batchsize):

                    if self.dim == 2:
                        vt = self.V
                    elif self.dim == 3:
                        vt = self.V[:, :, batch * batchsize + i]

                    # Forward
                    rt = self.visible_to_expected_hidden(vt, AF=AF) # E[h | v_data]

                    # Perform contrastive divergence and compute model statistics
                    if PCD and epoch != 0:
                        # use last gibbs sample as input (Persistant Contrastive Divergence)
                        barht, barvt, ht_k, vt_k = self.CD(vt_k[:, :, -1], rt, CDk, AF=AF)
                    else:
                        # use data (normal Contrastive Divergence)
                        barht, barvt, ht_k, vt_k = self.CD(vt, rt, CDk, AF=AF) # E[h | v_model]

                    self.vt_k = vt_k[:, :, -1]
                    self.barvt = barvt

                    # Accumulate error
                    err += torch.sum((vt - vt_k[:, :, -1]) ** 2)

                    # Backpropagation, compute gradients
                    dparam = self.grad(vt, rt, ht_k, vt_k, barvt, barht, CDk)
                    for i in range(len(dparam)): self.dparams[i] += dparam[i] / batchsize

                # Update gradients
                Dparams = self.update_grad(Dparams, lr=lrs[epoch], mom=mom, wc=wc, sp=sp, x=x)

            # hidden weights normalisation
            if U_normalisation and (epoch % 10) == 0:
                U = self.params[1]
                for i in range(self.N_H):
                    U[:, i] /= torch.mean(U[:, i])

            self.errors[epoch] = err / self.V.numel()

            if self.debug:
                self.parameter_history.append([param.detach().clone() for param in self.params])

    def return_params(self):
        return [self.W, self.U, self.b_V, self.b_init, self.b_H, self.errors]

    def CD(self, vt, rt, CDk, AF=torch.sigmoid):

        """     Performs the contrastive divergence step.

            Parameters
            ----------
            vt : torch.Tensor
                The input data.
            rt : torch.Tensor
                The expected hidden layer.
            CDk: int
                Number of constrastive divergence steps
            AF : function
                The activation function.

            Returns
            -------
            barht : torch.Tensor
                The mean over the CDks steps of the hidden layer activations of the model.
            barvt : torch.Tensor
                The mean over the CDks steps of the visible layer activations of the model.
            ht_k : torch.Tensor
                The hidden layer activations of the model for each CD step.
            vt_k : torch.Tensor
                The visible layer activations of the model for each CD step.

        """

        ht_k = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        probht_k = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        vt_k = torch.zeros(self.N_V, self.T, CDk, dtype=self.dtype, device=self.device)

        vt_k[:, :, 0] = vt.detach()
        probht_k[:, :, 0], ht_k[:, :, 0] = self.visible_to_hidden(vt_k[:, :, 0], rt, AF=AF)

        for kk in range(1, CDk):
            vt_k[:, :, kk] = self.hidden_to_visible(ht_k[:, :, 0], AF=AF)
            probht_k[:, :, kk], ht_k[:, :, kk] = self.visible_to_hidden(vt_k[:, :, kk], rt, AF=AF)

        barht = torch.mean(probht_k, 2)
        barvt = torch.mean(vt_k, 2)

        return barht, barvt, ht_k, vt_k

    def visible_to_expected_hidden(self, vt, AF=torch.sigmoid):

        """     Computes the expected hidden layer activations of the RTRBM.

            Parameters
            ----------
            vt : torch.Tensor
                The input data.
            AF : function
                The activation function.

            Returns
            -------
            rt : torch.Tensor
                The expected hidden layer activations.

        """

        T = vt.shape[1]
        rt = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        rt[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, T):
            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))
        return rt

    def visible_to_hidden(self, vt, r, AF=torch.sigmoid):

        """     Computes the hidden layer activations of the RTRBM.

            Parameters
            ----------
            vt : torch.Tensor
                The input data.
            r  : torch.Tensor
                The expected hiddens of the input data
            AF : function
                The activation function.

            Returns
            -------
            ph_sample : torch.Tensor
                The expectation (or probability hi=1) of the hidden layer activations.
        """

        T = vt.shape[1]
        ph_sample = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        ph_sample[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        ph_sample[:, 1:T] = AF(
            torch.matmul(self.W, vt[:, 1:T]).T + torch.matmul(self.U, r[:, 0:T - 1]).T + self.b_H).T
        return ph_sample, torch.bernoulli(ph_sample)

    def hidden_to_visible(self, h, AF=torch.sigmoid):

        """     Computes the visible layer activations of the RTRBM.

            Parameters
            ----------
            h : torch.Tensor
                The hidden layer activation.
            AF : function
                The activation function.

            Returns
            -------
            vt : torch.Tensor
                The visible layer activations.
        """

        return torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

    def initialize_grad_updates(self):
        """
        Initializes a list of zero tensors with the same shape as the model parameters.

        Parameters:
            self (torch.nn.Module): The model.

        Returns:
            list: A list of zero tensors with the same shape as the model parameters.
        """

        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad(self, vt, rt, ht_k, vt_k, barvt, barht, CDk):

        """     Computes the gradient of the parameters of the RTRBM.
                Parameters

                    Parameters
                    ----------
                     vt : torch.Tensor
                         The visible layer activations.
                     rt : torch.Tensor
                         The hidden layer activations.
                   ht_k : torch.Tensor
                          The hidden layer activations of the model for each CD step.
                   vt_k : torch.Tensor
                          The visible layer activations of the model for each CD step
                  barht : torch.Tensor
                          The mean over the CDks steps of the hidden layer activations of the model.
                  barvt : torch.Tensor
                          The mean over the CDks steps of the visible layer activations of the model.
                    CDk : int
                          The number of Gibbs sampling steps used to compute the negative phase in CD-k

                    Returns
                    -------

                     A list containing the gradient of the parameters of the RTRBM
        """

        Dt = torch.zeros(self.N_H, self.T + 1, dtype=self.dtype, device=self.device)

        for t in range(self.T - 1, -1, -1):  # begin, stop, step
            Dt[:, t] = torch.matmul(self.U.T, (Dt[:, t + 1] * rt[:, t] * (1 - rt[:, t]) + (rt[:, t] - barht[:, t])))

        db_init = (rt[:, 0] - barht[:, 0]) + Dt[:, 1] * rt[:, 0] * (1 - rt[:, 0])

        tmp = torch.sum(Dt[:, 2:self.T] * (rt[:, 1:self.T - 1] * (1 - rt[:, 1:self.T - 1])), 1)
        db_H = torch.sum(rt[:, 1:self.T], 1) - torch.sum(barht[:, 1:self.T], 1) + tmp

        db_V = torch.sum(vt - barvt, 1)

        dW_1 = torch.sum(
            (Dt[:, 1:self.T] * rt[:, 0:self.T - 1] * (1 - rt[:, 0:self.T - 1])).unsqueeze(1).repeat(1, self.N_V, 1) *
            vt[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)
        dW_2 = torch.sum(rt.unsqueeze(1).repeat(1, self.N_V, 1) * vt.unsqueeze(0).repeat(self.N_H, 1, 1), 2) - \
               torch.sum(
                   torch.sum(ht_k.unsqueeze(1).repeat(1, self.N_V, 1, 1) * vt_k.unsqueeze(0).repeat(self.N_H, 1, 1, 1),
                             3), 2) / CDk
        dW = dW_1 + dW_2

        dU = torch.sum((Dt[:, 2:self.T + 1] * (rt[:, 1:self.T] * (1 - rt[:, 1:self.T])) + rt[:, 1:self.T] - barht[:,
                                                                                                                1:self.T]).unsqueeze(
            1).repeat(1, self.N_H, 1) *
                           rt[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)

        del Dt

        return [dW, dU, db_H, db_V, db_init]

    def update_grad(self, Dparams, lr=1e-3, mom=0, wc=0, x=2, sp=None):

        """     Updates the parameters of the RTRBM with the gradients.

                    Parameters
                    ----------
                Dparams : List
                          A list containing the gradient of the parameters of the RTRBM in type torch.tensor
                     lr : float
                          learning rate of gradient decent
                   mom  : float
                          Momentum of gradient decent
                     wc : float
                          Weightcost of gradient decent
                      x : int
                          Polynomial proportionality of the sparse penalty relative to the weights
                     sp : float
                          Sparse penalty on the visible to hidden weight

                    Returns
                    -------

                     A list containing the gradient of the parameters of the RTRBM updates with the
                     learning parameters of gradient decent
        """

        dW, dU, db_H, db_V, db_init = self.dparams
        DW, DU, Db_H, Db_V, Db_init = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Db_H = mom * Db_H + lr * db_H
        Db_V = mom * Db_V + lr * db_V
        Db_init = mom * Db_init + lr * db_init

        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V), \
                                     [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, DU, Db_H, Db_V, Db_init]

        for i in range(len(self.params)): self.params[i] += Dparams[i]

        return [DW, DU, Db_H, Db_V, Db_init]


    def sample(self,
               v_start,
               AF=torch.sigmoid,
               chain=50,
               pre_gibbs_k=100,
               gibbs_k=20,
               mode=1,
               disable_tqdm=False):

        """     Infers data from the distribution of a trained RTRBM.

                    Parameters
                    ----------
                v_start : torch.Tensor
                          Starting position for inferring the data
                     AF : function
                          The activation function.
                 chain  : int
                          Number of Monte Carlo chains
             pre_gibbs_k : int
                          Burn in contrastive divergence
                gibbs_k : int
                          Number of contrastive divergence steps before computing the next
                          time step data points
                    mode : int
                          Three possible mode's:
                          1 -> take last gibbs sample (most often used)
                          2 -> take the mean over gibbs samples
                          3 -> take the most probable gibbs sample according to the free energy

                    Returns
                    -------

                     vt  : torch.Tensor
                           Inferred visibles
                     rt  : torch.Tensor
                           Inferred hiddens
        """

        vt = torch.zeros(self.N_V, chain+1, dtype=self.dtype, device=self.device)
        rt = torch.zeros(self.N_H, chain+1, dtype=self.dtype, device=self.device)

        rt[:, 0] = AF(torch.matmul(self.W, v_start.T) + self.b_init)
        vt[:, 0] = v_start
        for t in tqdm(range(1, chain+1), disable=disable_tqdm):
            v = vt[:, t - 1]

            # it is important to keep the burn-in inside the chain loop, because we now have time-dependency
            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_H + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

            vt_k = torch.zeros(self.N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_H + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_V, vt_k) + torch.matmul(
                    self.b_H, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))

        return vt[:, 1:], rt[:, 1:]

# EXAMPLE RUN 1:
import seaborn as sns
import numpy as np
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt


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



"""
# Generate bouncing ball data
N_V, N_H, T = 16, 8, 64
data = create_BB(N_V=N_V, T=T, n_samples=64, width_vec=[4, 5, 6, 7], velocity_vec=[2, 3], boundary=False)
data = data.to('cpu')
sns.heatmap(data[:, :, 0])
plt.xlabel('time')
plt.ylabel('Neuron index')
plt.title('Training data one example batch')

# Initialize and train RTRBM
rtrbm = RTRBM(data, N_H=N_H, device="cpu", debug=True)
rtrbm.learn(batchsize=64, n_epochs=100, lr=1e-2, lr_end=1e-3, start_decay=50, sp=1e-5, x=2)

# Infer from trained RTRBM and plot some results
vt_infer, rt_infer = rtrbm.sample(torch.tensor(data[:, T//2, 0], dtype=torch.float))

_, ax = plt.subplots(2, 2, figsize=(12, 12))
sns.heatmap(vt_infer.cpu(), ax=ax[0, 0], cbar=False)
ax[0, 0].set_title('Infered data')
ax[0, 0].set_xlabel('Time')
ax[0, 0].set_ylabel('Neuron index')

ax[0, 1].plot(rtrbm.errors.cpu())
ax[0, 1].set_title('RMSE of the RTRBM over epoch')
ax[0, 1].set_xlabel('Epoch')
ax[0, 1].set_ylabel('RMSE')

sns.heatmap(rtrbm.W.cpu(), ax=ax[1, 0])
ax[1, 0].set_title('Visible to hidden connection')
ax[1, 0].set_xlabel('Visible')
ax[1, 0].set_ylabel('Hiddens')

sns.heatmap(rtrbm.U.cpu(), ax=ax[1, 1])
ax[1, 1].set_title('Hidden to hidden connection')
ax[1, 1].set_xlabel('Hidden(t-1)')
ax[1, 1].set_ylabel('Hiddens(t)')
plt.show()



# EXAMPLE RUN 2

def artificial_neuronal_data(neurons_per_pop=40, n_pop=3, n_batches=300, T=100, delay=1):
    '''
    Creates artificial neuronal data of 6 interacting neural assemblies, that each receive an external
    input firing rate in the form of a random sinusoid

    :param neurons_per_pop: Number of neurons per neural assembly
    :param n_pop: Number of neural assemblies NOT adjustable yet
    :param n_batches: Number of batches
    :param T: Duration of one batch
    :param delay: Delay of interaction between neural assemblies
    :return: neural activity of the whole population
    '''

    ######## Defining coordinate system ########
    rads = torch.linspace(0, 2*torch.pi, n_pop+1)
    mean_locations_pop = torch.zeros(n_pop, 2)
    coordinates = torch.zeros(neurons_per_pop*n_pop, 2)
    for i in range(n_pop):
        mean_locations_pop[i, :] = torch.tensor([torch.cos(rads[i]), torch.sin(rads[i])])
        coordinates[neurons_per_pop * i:neurons_per_pop * (i + 1), :] = 0.15 * torch.randn(neurons_per_pop, 2) + mean_locations_pop[i]

    ######## Start creating data ########
    data = torch.zeros(neurons_per_pop*n_pop, T, n_batches)
    for batch in tqdm(range(n_batches)):

        ######## Creating random input currents and mother trains ########
        t = np.linspace(0, 1*np.pi, T+delay)
        fr = np.zeros((n_pop, T+delay))
        mother = np.zeros((n_pop, T+delay))
        freq_m = np.random.randint(low=500, high=750)
        for pop in range(n_pop):
            u = np.random.rand()
            phase = freq_m * np.random.randn()
            amp = .5 * np.random.rand()
            shift = .68*np.random.rand()
            fr[pop, :] = amp*np.sin(phase*(t + 2*np.pi*u)) + shift
            while np.min(fr[pop, :]) < 0:
                u = np.random.rand()
                phase = freq_m * np.random.randn()
                amp = .1*np.random.rand()
                shift = .68*np.random.rand()
                fr[pop, :] = amp*np.sin(phase*(t + 2*np.pi*u)) + shift
            mother[pop, :] = np.random.poisson(fr[pop, :])


        # empty data array
        spikes = np.zeros((neurons_per_pop*n_pop, T+delay))

        # Excitatory and inhibitory connections
        inh = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        exc = [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
        for pop in range(n_pop):
            delete_spikes = np.roll(np.sum(fr[inh[pop], :], 0), delay) * np.ones((neurons_per_pop, T+delay)) >= np.random.uniform(0, 1, size=(neurons_per_pop, T+delay))
            noise = np.random.poisson(np.roll(np.sum(fr[exc[pop], :], 0), delay), (neurons_per_pop, T+delay))
            temp = np.tile(mother[pop, :], (neurons_per_pop, 1)) - delete_spikes + noise
            spikes[pop*neurons_per_pop:(pop+1)*neurons_per_pop, :] = temp[np.argsort(np.mean(temp, 1)), :]
        spikes[spikes < 0] = 0
        spikes[spikes > 1] = 1
        data[:, :, batch] = torch.tensor(spikes[:, delay:])
    #print(torch.mean(data))

    return data, coordinates

# Generate artificial neuronal data
data, coordinates = artificial_neuronal_data()
sns.heatmap(data[:, :, 0])
plt.xlabel('time')
plt.ylabel('Neuron index')
plt.title('Training data one example batch')

# Initialze and train RTRBM
rtrbm = RTRBM(data, N_H=6, device="cpu")
rtrbm.learn(batchsize=50, n_epochs=500, lr=1e-2, lr_end=1e-3, start_decay=50, sp=1e-5, x=2)

# Infer from trained RTRBM and plot some results
vt_infer, rt_infer = rtrbm.sample(torch.tensor(data[:, 100//2, 0], dtype=torch.float))

_, ax = plt.subplots(2, 2, figsize=(12, 12))
sns.heatmap(vt_infer.cpu(), ax=ax[0, 0], cbar=False)
ax[0, 0].set_title('Infered data')
ax[0, 0].set_xlabel('Time')
ax[0, 0].set_ylabel('Neuron index')

ax[0, 1].plot(rtrbm.errors.cpu())
ax[0, 1].set_title('RMSE of the RTRBM over epoch')
ax[0, 1].set_xlabel('Epoch')
ax[0, 1].set_ylabel('RMSE')

sns.heatmap(rtrbm.W.cpu(), ax=ax[1, 0])
ax[1, 0].set_title('Visible to hidden connection')
ax[1, 0].set_xlabel('Visible')
ax[1, 0].set_ylabel('Hiddens')

sns.heatmap(rtrbm.U.cpu(), ax=ax[1, 1])
ax[1, 1].set_title('Hidden to hidden connection')
ax[1, 1].set_xlabel('Hidden(t-1)')
ax[1, 1].set_ylabel('Hiddens(t)')
plt.show()
"""