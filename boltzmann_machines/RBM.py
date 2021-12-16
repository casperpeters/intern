'Scipt for the RTM'

import torch
from tqdm import tqdm

x=1
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
            self.data = data.float().to(self.device)
            self.N_V, self.num_samples = self.data.shape
            self.dim = torch.tensor(self.data.shape).shape[0]
        else:
            raise ValueError("Data is not correctly defined: Use (N_V) or (N_V, num_samples) dimensions.\
                             If you want to have (N_V, T, num_samples) try to reshape it to (N_V, T*num_samples).\
                             And if you want to train on each sample separately set batchsize=T.")

        self.N_H = N_H

        self.W = 0.01 * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.b_H = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(self.N_V, dtype=self.dtype, device=self.device)

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
            num_batches = self.num_samples // batchsize # same as floor // (round to the bottom)

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

                for i in range(0, batchsize):
                    if self.dim == 1:
                        v = self.data
                    elif self.dim == 2:
                        v = self.data[:, batch * batchsize + i]

                    # Perform contrastive divergence and compute model statistics
                    if PCD and epoch != 0:
                        # use last gibbs sample as input (Persistant Contrastive Divergence)
                        vk, pvk, hk, phk, ph, h = self.CD(vk, CDk, AF=AF)
                        ph, h = self.visible_to_hidden(v, AF=AF)
                    else:
                        # use data (normal Contrastive Divergence)
                        vk, pvk, hk, phk, ph, h = self.CD(v, CDk, AF=AF)

                    # Accumulate error
                    err += torch.sum((v - vk) ** 2)

                    # Backpropagation, compute gradients
                    dparam = self.grad(v, vk, ph, phk)
                    for j in range(len(dparam)): self.dparams[j] += dparam[j] / batchsize

                # Update gradients
                Dparams = self.update_grad(Dparams, lr=lr, mom=mom, wc=wc, sp=sp, x=x)

            self.errors[epoch] = err / self.data.numel()

            if lr and lr_end and start_decay is not None:
                if start_decay <= epoch:
                    lr *= r

    def CD(self, v, CDk, AF=torch.sigmoid):

        ph, h = self.visible_to_hidden(v)
        hk = h.detach().clone()
        for k in range(CDk):
            pvk, vk = self.hidden_to_visible(hk, AF=AF)
            phk, hk = self.visible_to_hidden(vk, AF=AF)

        return vk, pvk, hk, phk, ph, h

    def visible_to_hidden(self, v, AF=torch.sigmoid):

        p = AF(torch.matmul(v, self.W.T) + self.b_H)

        return p, torch.bernoulli(p)

    def hidden_to_visible(self, h, AF=torch.sigmoid):

        p = AF(torch.matmul(h, self.W) + self.b_V)

        return p, torch.bernoulli(p)

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad(self, v, vk, ph, phk):

        dW = torch.outer(ph, v.T) - torch.outer(phk, vk.T)
        db_V = v - vk
        db_H = ph - phk

        return [dW, db_H, db_V]

    def update_grad(self, Dparams, lr=1e-3, mom=0, wc=0, sp=None, x=2):

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

    def free_energy(self, v):

        v_term = torch.outer(v, self.b_V.T)
        w_x_h = torch.nn.functional.linear(v, self.W.T, self.b_H)
        h_term = torch.sum(torch.nn.functional.softplus(w_x_h))

        return torch.mean(-h_term - v_term)

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

        v = v_start

        for kk in range(pre_gibbs_k):
            _, h = self.visible_to_hidden(v, AF=AF)
            _, v = self.hidden_to_visible(h, AF=AF)

        for t in tqdm(range(chain), disable=disable_tqdm):
            vt_k = torch.zeros(self.N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                _, h = self.visible_to_hidden(v, AF=AF)
                _, v = self.hidden_to_visible(h, AF=AF)
                vt_k[:, kk] = v
                ht_k[:, kk] = h

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
                ht[:, t] = ht_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)

        return vt, ht

"""
# This is an example run:
from data.reshape_data import *
from data.mock_data import *
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt

N_V, N_H, T = 16, 8, 64
data = create_BB(N_V=N_V, T=T, n_samples=24, width_vec=[4, 5, 6, 7], velocity_vec=[2, 3], boundary=False)
mean = torch.zeros(N_H, 20)
std = torch.zeros(N_H, 20)

rbm = RBM(data, N_H=N_H, device="cpu")
rbm.learn(batchsize=6, n_epochs=100, lr=1e-3, mom=0, wc=0, sp=None, disable_tqdm=False)

vt_infer, rt_infer = rbm.sample(torch.tensor(data[:, T//2, 0], dtype=torch.float))

sns.heatmap(vt_infer.cpu())
plt.show()

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
"""