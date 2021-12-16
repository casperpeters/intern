'Scipt for the RTM'

import torch
from tqdm import tqdm


class RBM(object):

    def __init__(self, data, N_H=10, device=None):

        if device is None:
            self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        else:
            self.device = device

        self.dtype = torch.float
        self.data = data.float().to(self.device)
        self.dim = torch.tensor(self.data.shape).shape[0]

        if self.dim == 1:
            self.N_V = self.data.shape
        elif self.dim == 2:
            self.N_V, self.num_samples = self.data.shape
        elif self.dim == 3:
            N_V, T, num_samples = data.shape
            data = torch.zeros(N_V, T * num_samples)
            for i in range(num_samples):
                data[:, T * i:T * (i + 1)] = self.data[:, :, i]
            self.data = data
            self.N_V, self.num_samples = self.data.shape
            self.dim = torch.tensor(self.data.shape).shape[0]
        else:
            raise ValueError("Data is not correctly defined: Use (N_V) or (N_V, num_samples) dimensions.\
                             If you want to have (N_V, T, num_samples) try to reshape it to (N_V, T*num_samples).\
                             And if you want to train on each sample separately set batchsize=T.")

        self.N_H = N_H
        self.W = 0.01 * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)

        self.mu_H = 0.5 * torch.ones(self.N_H, 1, dtype=self.dtype, device=self.device)
        self.mu_V = torch.tensor(torch.mean(self.data, 1), dtype=self.dtype, device=self.device)[:, None]

        # initialize bias such that p(v=1|h)=sigmoid(logit(<v>_data))=<v>_data
        self.b_H = -torch.log(1 / self.mu_H - 1)
        self.b_V = -torch.log(1 / self.mu_V - 1)

        self.params = [self.W, self.b_H, self.b_V]
        self.dparams = self.initialize_grad_updates()

    def learn(self,
              n_epochs=1000,
              batchsize=1,
              CDk=10, PCD=False,
              lr=1e-3, lr_end=None, start_decay=None,
              sp=None, x=2,
              mom=0.9,
              wc=0.0002,
              AF=torch.sigmoid,
              disable_tqdm=False):

        global vt_k
        if self.dim == 1:
            num_batches = 1
            batchsize = 1
        elif self.dim == 2:
            num_batches = self.num_samples // batchsize  # same as floor // (round to the bottom)

        # learing rate
        if lr and lr_end and start_decay is not None:
            r = (lr_end / lr) ** (1 / (n_epochs - start_decay))

        Dparams = self.initialize_grad_updates()

        self.errors = torch.zeros(n_epochs, 1, dtype=self.dtype, device=self.device)
        self.disable = disable_tqdm

        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0

            for batch in range(num_batches):
                self.dparams = self.initialize_grad_updates()

                if self.dim == 1:
                    v_data = self.data
                elif self.dim == 2:
                    v_data = self.data[:, batch * batchsize: (batch + 1) * batchsize]

                if PCD and epoch != 0:
                    # use last gibbs sample as input (Persistant Contrastive Divergence)
                    v_model, h_model_mean, h_data_mean = self.CD(v_model, CDk, AF=AF)
                    ph, h = self.visible_to_hidden(v_data, AF=AF)
                else:
                    # use data (normal Contrastive Divergence)
                    v_model, h_model, h_data = self.CD(v_data, CDk, AF=AF)
                # Accumulate error
                err += torch.sum((v_data - v_model) ** 2)

                mu_V_batch = torch.mean(v_data, 1)[:, None]
                mu_H_batch = torch.mean(h_data, 1)[:, None]

                # Translate biases with respect to the new offsets and update offsets
                self.translate_fields_and_update_offsets(mu_H_batch, mu_V_batch, num_batches, batch)

                # Compute mean gradients of batch
                dparam = self.grad(v_data, v_model, h_data, h_model)
                for j in range(len(dparam)): self.dparams[j] += dparam[j] / batchsize

            self.errors[epoch] = err / self.data.numel()

            if lr and lr_end and start_decay is not None:
                if start_decay <= epoch:
                    lr *= r

    def weight_divergence(self, lr, Dparams_min, Dparams):
        if torch.std(torch.abs(self.W - self.W_min)) > lr:
            lr = lr/10
            Dparams = self.compute_batch_grad_and_update_grad(Dparams_min)
        self.W_min = self.W
        return lr, Dparams

    def CD(self, v_data, CDk, AF=torch.sigmoid):
        ''' Function that performs the Contrastive Divergence algorithm by sequentially sampling from the
            visible layer to the hidden layer and back

            param x_data_mean:  P(h=1|v_data) V P(v=1|h_data)
            param x_model_mean: P(h=1|v_model) V P(v=1|h_model)
            param x_data:       Bernoulli(x_data_mean)
            param x_model:      Bernoulli(x_model_mean)
        '''

        h_data_mean, h_data = self.visible_to_hidden(v_data)
        h_model = h_data.detach().clone()
        for k in range(CDk):
            v_model_mean, v_model = self.hidden_to_visible(h_model, AF=AF)
            h_model_mean, h_model = self.visible_to_hidden(v_model, AF=AF)

        return v_model, h_model_mean, h_data_mean

    def visible_to_hidden(self, v, AF=torch.sigmoid):
        ''' Function to compute the transition from the visible to hidden layer
            param p: P(v=1|h)
        '''

        p = AF(torch.matmul(self.W, (v - self.mu_V)) + self.b_H)

        return p, torch.bernoulli(p)

    def hidden_to_visible(self, h, AF=torch.sigmoid):
        ''' Function to compute the transition from the hidden to visible layer
            param p: P(h=1|v)
        '''

        p = AF(torch.matmul(self.W.T, (h - self.mu_H)) + self.b_V)

        return p, torch.bernoulli(p)

    def initialize_grad_updates(self):
        '''This function initialize each parameter defined in the list self.params to zero'''
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad(self, v_data, v_model, h_data, h_model):
        ''' This function computes the gradient of a batch with the centering trick

            param dx: mean gradient of a batch
        '''
        dW = torch.matmul((h_data - self.mu_H), (v_data - self.mu_V).T) -\
             torch.matmul((h_model - self.mu_H), (v_model - self.mu_V).T)

        if self.dim == 2:
            db_V = torch.sum(v_data - v_model, 1)[:, None]
            db_H = torch.sum(h_data - h_model, 1)[:, None]
        elif self.dim == 1:
            db_V = v_data - v_model
            db_H = h_data - h_model

        return [dW, db_H, db_V]

    def compute_batch_grad_and_update_grad(self, Dparams, lr=1e-3, mom=0, wc=0, sp=None, x=2):
        ''' This function computes the mean batch gradients, implements the sparse penalty (sp), and updates
            the global gradients

            param dx: mean gradient of a batch
            param Dx: gradient with implemented momentum (mom), weightcost (wc), sparse penalty and centering trick
        '''
        dW, db_H, db_V = self.dparams
        DW, Db_H, Db_V = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        Db_H = mom * Db_H + lr * db_H
        Db_V = mom * Db_V + lr * db_V

        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V),
                                     [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, Db_H, Db_V]

        for i in range(len(self.params)): self.params[i] += Dparams[i]

        return Dparams

    def translate_fields_and_update_offsets(self, mu_H_batch, mu_V_batch, num_batches, batch):
        ''' This function translate the model biases with respect to their new offsets'''
        lr_offset = 0.1 * torch.logspace(1, -2, num_batches, base=10, dtype=self.dtype, device=self.device)[batch]
        self.b_H += lr_offset * torch.matmul(self.W, (mu_V_batch - self.mu_V))
        self.b_V += lr_offset * torch.matmul(self.W.T, (mu_H_batch - self.mu_H))
        self.mu_H = (1 - lr_offset) * self.mu_H + lr_offset * mu_H_batch
        self.mu_V = (1 - lr_offset) * self.mu_V + lr_offset * mu_V_batch
        return

    def free_energy(self, v):
        ''' This function computes the free energy of the RBM'''

        v_term = torch.outer(v, self.b_V.T)
        w_x_h = torch.nn.functional.linear(v, self.W.T, self.b_H)
        h_term = torch.sum(torch.nn.functional.softplus(w_x_h))

        return torch.mean(-h_term - v_term)

    def batch_parameters(self, batch, batchsize):
        ''' This function returns the visibles of the batch, and predefine empty tensors for the sampled model visibles,
            hiddens of the data and the hiddens of the model
        '''
        v_data = self.data[:, batch:batch + batchsize]
        v_model = torch.zeros(self.N_V, batchsize)
        h_data = torch.zeros(self.N_H, batchsize)
        h_model = torch.zeros(self.N_H, batchsize)
        return v_data, v_model, h_data, h_model

    def sample(self,
               v_start,
               AF=torch.sigmoid,
               pre_gibbs_k=100,
               gibbs_k=20,
               mode=1,
               chain=50,
               disable_tqdm=False):

        vt = torch.zeros(self.N_V, chain, dtype=self.dtype, device=self.device)
        ht = torch.zeros(self.N_H, chain, dtype=self.dtype, device=self.device)

        v = v_start[:, None]

        for kk in range(pre_gibbs_k):
            _, h = self.visible_to_hidden(v, AF=AF)
            _, v = self.hidden_to_visible(h, AF=AF)

        for t in tqdm(range(chain), disable=disable_tqdm):
            vt_k = torch.zeros(self.N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                _, h = self.visible_to_hidden(v, AF=AF)
                _, v = self.hidden_to_visible(h, AF=AF)
                vt_k[:, kk] = v[:, 0]
                ht_k[:, kk] = h[:, 0]

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
                ht[:, t] = ht_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)

        return vt, ht


# This is an example run:

# This is an example run:
from data.reshape_data import *
from data.mock_data import *
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt

N_V, N_H, T = 16, 8, 64
data = create_BB(N_V=N_V, T=T, n_samples=256, width_vec=[4, 5, 6, 7], velocity_vec=[2, 3], boundary=False)
mean = torch.zeros(N_H, 20)
std = torch.zeros(N_H, 20)

rbm = RBM(data, N_H=N_H, device="cpu")
rbm.learn(batchsize=1, n_epochs=100, lr=1e-3)#, lr_end=1e-5, start_decay=100)

#vt_infer, rt_infer = rbm.sample(torch.tensor(data[:, T//2, 0], dtype=torch.float))

#sns.heatmap(vt_infer.cpu())
#plt.show()

plt.plot(rbm.errors.cpu())
plt.show()

sns.heatmap(rbm.W)
plt.show()

from utils.plots import *
N_V, T, num_samples = data.shape
data1 = torch.zeros(N_V, T * num_samples)
for i in range(num_samples):
    data1[:, T * i:T * (i + 1)] = data[:, :, i]

rt, h = rbm.visible_to_hidden(data1)
plot_effective_coupling_VH(rbm.W, data1.float(), rt.float())
