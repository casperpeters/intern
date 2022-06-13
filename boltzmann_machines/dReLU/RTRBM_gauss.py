"""
Basic implementation of the recurrent temporal restricted boltzmann machine[1] with gaussian hidden units.

~ Sebastian Quiroz Monnens
References
----------
[1] : Sutskever, I., Hinton, G.E., & Taylor, G. (2008) The Recurrent Temporal Restricted Boltzmann Machine
"""

import torch
import numpy as np
from tqdm import tqdm
from optim.lr_scheduler import get_lrs


class RTRBM_G(object):
    def __init__(self, data: torch.Tensor, n_hidden: int = 10, device: str = 'cpu', debug_mode: bool = False):
        if not torch.cuda.is_available():
            print('cuda not available, using cpu')
            self.device = 'cpu'
        else:
            self.device = device
        self.V = data
        if self.V.ndim == 2:
            self.V = self.V[..., None]
        self.n_visible, self.T, self.num_samples = self.V.shape
        if self.V.ndim != 3:
            raise ValueError(
                "Data is not correctly defined: Use (n_visible, T) or (n_visible, T, num_samples) dimensions")

        self.n_hidden = n_hidden
        self.W = 0.01 / self.n_visible * torch.randn(self.n_hidden, self.n_visible, dtype=torch.float,
                                                     device=self.device)
        self.U = 0.01 / self.n_hidden * torch.randn(self.n_hidden, self.n_hidden, dtype=torch.float, device=self.device)
        self.b_v = torch.zeros(self.n_visible, dtype=torch.float, device=self.device)
        self.theta_0 = torch.zeros(self.n_hidden, dtype=torch.float, device=self.device)
        self.gamma_0 = torch.ones(self.n_hidden, dtype=torch.float, device=self.device)
        self.theta = torch.zeros(self.n_hidden, dtype=torch.float, device=self.device)
        self.gamma = torch.ones(self.n_hidden, dtype=torch.float, device=self.device) / (torch.var(data) + 1e-6)
        self.params = [self.W, self.U, self.theta, self.gamma, self.b_v, self.theta_0, self.gamma_0]

        self.debug_mode = debug_mode
        if debug_mode:
            self.parameter_history = []
        self.Dparams = self.initialize_grad_updates()
        self.errors = []

    def _parallel_recurrent_sample_r_given_v(self, v: torch.Tensor) -> torch.Tensor:
        _, T, n_samples = v.shape
        r = torch.zeros(self.n_hidden, T, n_samples, device=v.device)

        I0 = self.lnorm(torch.einsum('hv, vb -> hb', self.W, v[:, 0, :]))
        r[:, 0, :] = (I0 - self.theta_0[:, None]) / self.gamma_0[:, None]

        for t in range(1, T):
            I = self.lnorm(torch.einsum('hv, vb -> hb', self.W, v[:, t, :]) +
                                torch.einsum('hr, rb -> hb', self.U, r[:, t - 1, :]))
            r[:, t, :] = (I - self.theta[:, None]) / self.gamma[:, None]
        return r

    def _parallel_sample_h_given_v(self, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        _, T, n_samples = v.shape

        I0 = self.lnorm(torch.einsum('hv, vb -> hb', self.W, v[:, 0, :]))
        mean_0 = (I0 - self.theta_0[:, None]) / self.gamma_0[:, None]
        std_0 = 1 / torch.sqrt(self.gamma_0[:, None].repeat(1, n_samples))

        I = self.lnorm(torch.einsum('hv, vTb->hTb', self.W, v[:, 1:, :]) + torch.einsum('hr, rTb->hTb', self.U, r[:, :-1, :]))
        mean = (I - self.theta[:, None, None]) / self.gamma[:, None, None]
        std = 1 / torch.sqrt(self.gamma[:, None, None].repeat(1, T - 1, n_samples))

        mean = torch.cat([mean_0[:, None, :], mean], 1)
        std = torch.cat([std_0[:, None, :], std], 1)
        if torch.sum(torch.isnan(std)) > 0:
            a = 1
        if torch.sum(torch.isinf(std)) > 0:
            a = 1
        h = torch.normal(mean=mean, std=std)
        return h

    def _parallel_sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        v_prob = torch.sigmoid(torch.einsum('hv, hTb->vTb', self.W, h) + self.b_v[:, None, None])
        if torch.sum(torch.isnan(v_prob)) > 0:
            a = 1
        if torch.sum(torch.isnan(torch.abs(v_prob))) > 0:
            a = 1
        return torch.bernoulli(v_prob)

    def CD(self, v_data: torch.Tensor, r_data: torch.Tensor, CDk: int = 1) -> torch.Tensor:
        _, T, n_samples = v_data.shape
        ht_k = torch.zeros(self.n_hidden, T, n_samples, CDk, dtype=torch.float, device=self.device)
        vt_k = torch.zeros(self.n_visible, T, n_samples, CDk, dtype=torch.float, device=self.device)
        vt_k[..., 0] = v_data.detach().clone()
        ht_k[..., 0] = self._parallel_sample_h_given_v(v_data, r_data)
        for i in range(1, CDk):
            vt_k[..., i] = self._parallel_sample_v_given_h(ht_k[..., i - 1])
            ht_k[..., i] = self._parallel_sample_h_given_v(vt_k[..., i], r_data)
        return torch.mean(ht_k, 3), torch.mean(vt_k, 3), ht_k, vt_k

    def learn(self, n_epochs=1000,
              lr=None,
              lr_schedule=None,
              batch_size=1,
              CDk=10,
              PCD=False,
              sp=None, x=2,
              mom=0.9, wc=0.0002,
              disable_tqdm=False,
              save_every_n_epochs=1, shuffle_batch=True, layer_norm=True, n=1,
              **kwargs):

        self.layer_norm = layer_norm
        self.n = n
        if self.num_samples <= batch_size:
            batch_size, num_batches = self.num_samples, 1
        else:
            num_batches = self.num_samples // batch_size

        if lr is None:
            lrs = np.array(get_lrs(mode=lr_schedule, n_epochs=n_epochs, **kwargs))
        else:
            lrs = lr * torch.ones(n_epochs)

        self.disable = disable_tqdm
        self.lrs = lrs
        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0
            for batch in range(0, num_batches):
                self.dparams = self.initialize_grad_updates()
                v_data = self.V[:, :, batch * batch_size: (batch + 1) * batch_size].to(self.device)
                r_data = self._parallel_recurrent_sample_r_given_v(v_data)

                if PCD and epoch != 0:
                    barht, barvt, ht_k, vt_k = self.CD(vt_k[:, :, -1], CDk)
                else:
                    barht, barvt, ht_k, vt_k = self.CD(v_data, r_data, CDk)

                err += torch.sum((v_data - vt_k[..., -1]) ** 2).cpu()
                self.dparams = self.grad(v_data, r_data, ht_k, vt_k, barvt, barht)

                self.update_grad(lr=lrs[epoch], mom=mom, wc=wc, sp=sp, x=x)

            self.errors += [err / self.V.numel()]
            if self.debug_mode and epoch % save_every_n_epochs == 0:
                self.parameter_history.append([param.detach().clone().cpu() for param in self.params])
            if shuffle_batch:
                self.V[..., :] = self.V[..., torch.randperm(self.num_samples)]

    def lnorm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if not self.layer_norm:
            return
        if x.ndim == 2 :# x.shape = n_h, T=0, n_batches
            return (x - torch.mean(x, 0)[None, :]) / (torch.std(x, 0)[None, :] + eps)
        elif x.ndim == 3:# x.shape = n_h, T-1, n_batches
            return ((x - torch.mean(x, 0)[None, :, :]) / (torch.std(x, 0)[None, :, :] + eps))

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=torch.float, device=self.device) for param in self.params]

    def grad(self, v_data: torch.Tensor, r_data: torch.Tensor, ht_k: torch.Tensor, vt_k: torch.Tensor,
             barvt: torch.Tensor, barht: torch.Tensor) -> torch.Tensor:
        T = self.T

        I0_data = self.lnorm(torch.einsum('hv, vb -> hb', self.W, v_data[:, 0, :]))
        I0_model = self.lnorm(torch.einsum('hv, vb -> hb', self.W, barvt[:, 0, :]))
        I_data = self.lnorm(torch.einsum('hv, vTb -> hTb', self.W, v_data[:, 1:, :]) + \
                            torch.einsum('hr, rTb->hTb', self.U, r_data[:, :-1, :]))
        I_model = self.lnorm(torch.einsum('hv, vTb -> hTb', self.W, barvt[:, 1:, :]) + \
                             torch.einsum('hr, rTb->hTb', self.U, barht[:, :-1, :]))

        std = torch.std(torch.cat([torch.einsum('hv, vb -> hb', self.W, v_data[:, 0, :])[:, None, :]
                                      ,torch.einsum('hv, vTb -> hTb', self.W, v_data[:, 1:, :]) + \
                            torch.einsum('hr, rTb->hTb', self.U, r_data[:, :-1, :])], 1), 0)[None, :, :]
        std[std==0] = 1e-6

        Dt = torch.zeros(self.n_hidden, T + 1, v_data.shape[2], dtype=torch.float, device=self.device)
        for t in range(T - 1, 0, -1):
            Dt[:, t, :] = torch.einsum('hv, hb->vb', self.U, (Dt[:, t + 1, :] * 1 / (std[:, t, :] * self.gamma[:, None]) + \
                                                              r_data[:, t, :] - barht[:, t, :]))

        Dt[:, 0, :] = torch.einsum('hv, hb->vb', self.U, (Dt[:, 1, :] * 1 / (std[:, 0, :] * self.gamma_0[:, None]) + \
                                                          r_data[:, 0, :] - barht[:, 0, :]))

        dtheta_0 = torch.mean(-I0_data + I0_model - Dt[:, 1, :], 1) / self.gamma_0

        dgamma_0 = torch.mean(-(I0_data - self.theta_0[:, None]) ** 2 + (I0_model - self.theta_0[:, None]) ** 2 - \
                              Dt[:, 1, :] * 2 * (I0_data - self.theta_0[:, None]), 1) / (2 * self.gamma_0 ** 2)

        dtheta = torch.mean(torch.sum(-I_data + I_model - Dt[:, 2:T+1, :], 1), 1) / self.gamma

        dgamma = torch.mean(
            torch.sum(-(I_data - self.theta[:, None, None]) ** 2 + (I_model - self.theta[:, None, None]) ** 2 - \
                      Dt[:, 2:T+1, :] * 2 * (I_data - self.theta[:, None, None]), 1), 1) / (2 * self.gamma ** 2)

        db_v = torch.mean(torch.sum(v_data - barvt, 1), 1)
        dW = torch.mean(
            torch.einsum('rTb, vTb -> rvb', r_data, v_data) - torch.mean(torch.einsum('rTbk, vTbk -> rvbk', ht_k, vt_k),
                                                                         3), 2)

        dW += torch.mean(torch.einsum('rb, vb -> rvb', Dt[:, 1, :] / (std[:, 0, :] * self.gamma_0[:, None]), v_data[:, 0, :]), 2)
        dW += torch.mean(torch.einsum('rTb, vTb -> rvb', Dt[:, 2:T, :] / (std[:, 1:-1, :] * self.gamma[:, None, None]), v_data[:, 1:- 1, :]), 2)
        dU = torch.mean(torch.einsum('rTb, hTb -> rhb', Dt[:, 2:T + 1, :] / (std[:, 1:, :] * self.gamma[:, None, None]) + \
                                     r_data[:, 1:T, :] - barht[:, 1:T, :], r_data[:, 0:T - 1, :]), 2)

        if torch.sum(torch.isnan(dtheta_0)) > 0:
            a = 1
        if torch.sum(torch.isnan(dgamma_0)) > 0:
            a = 1
        if torch.sum(torch.isnan(dgamma)) > 0:
            a = 1
        if torch.sum(torch.isnan(db_v)) > 0:
            a = 1
        if torch.sum(torch.isnan(dW)) > 0:
            a = 1
        if torch.sum(torch.isnan(dU)) > 0:
            a = 1

        return [dW, dU, dtheta, dgamma, db_v, dtheta_0, dgamma_0]

    def update_grad(self, lr=1e-3, mom=0, wc=0, sp=None, x=2):
        dW, dU, dtheta, dgamma, db_v, dtheta_0, dgamma_0 = self.dparams
        DW, DU, Dtheta, Dgamma, Db_v, Dtheta_0, Dgamma_0 = self.Dparams
        DW = mom * DW + lr/self.n * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Dtheta = mom * Dtheta + lr * dtheta
        Dgamma = mom * Dgamma + lr * dgamma
        Db_v = mom * Db_v + lr * db_v
        Dtheta_0 = mom * Dtheta_0 + lr * dtheta_0
        Dgamma_0 = mom * Dgamma_0 + lr * dgamma_0
        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.n_visible),
                                     [self.n_hidden, self.n_visible]) ** (x - 1) * torch.sign(self.W)

        self.Dparams = [DW, DU, Dtheta, Dgamma, Db_v, Dtheta_0, Dgamma_0]
        for i in range(len(self.params)): self.params[i] += self.Dparams[i]
        self.constraint()
        return

    def constraint(self, gamma_min=0.05, drop_max=0.75):
        self.gamma[self.gamma < gamma_min] = gamma_min
        self.gamma_0[self.gamma_0 < gamma_min] = gamma_min
        self.params = [self.W, self.U, self.theta, self.gamma, self.b_v, self.theta_0, self.gamma_0]
        return

    def return_params(self):
        return [self.W, self.U, self.theta, self.gamma, self.b_v, self.theta_0, self.gamma_0, self.errors]

    def infer(self,
              data,
              pre_gibbs_k=50,
              gibbs_k=10,
              mode=1,
              t_extra=0,
              disable_tqdm=False):

        T = self.T
        n_hidden = self.n_hidden
        if data.ndim == 2:
            data = data[..., None]
        n_visible, t1, n_samples = data.shape

        vt = torch.zeros(n_visible, T + t_extra, n_samples, dtype=torch.float, device=self.device)
        rt = torch.zeros(n_hidden, T + t_extra, n_samples, dtype=torch.float, device=self.device)

        vt[:, :t1, :] = data.float().to(self.device)
        rt[:, :t1, :] = self._parallel_recurrent_sample_r_given_v(vt[:, :t1, :])

        std = 1 / torch.sqrt(self.gamma[:, None].repeat(1, n_samples))
        for t in tqdm(range(t1, T + t_extra), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                I = self.lnorm(torch.einsum('hv, vb->hb', self.W, v) + torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]))
                mean = (I - self.theta[:, None]) / self.gamma[:, None]
                h = torch.normal(mean=mean, std=std)
                v = torch.bernoulli(torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_v[:, None]))

            vt_k = torch.zeros(n_visible, gibbs_k, n_samples, dtype=torch.float, device=self.device)
            ht_k = torch.zeros(n_hidden, gibbs_k, n_samples, dtype=torch.float, device=self.device)
            for kk in range(gibbs_k):
                I = self.lnorm(torch.einsum('hv, vb->hb', self.W, v) + torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]))
                mean = (I - self.theta[:, None]) / self.gamma[:, None]
                h = torch.normal(mean=mean, std=std)
                v = torch.bernoulli(torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_v[:, None]))
                vt_k[:, kk, :] = v.detach().clone()
                ht_k[:, kk, :] = h.detach().clone()

            if mode == 1:
                vt[:, t, :] = vt_k[:, -1, :]
            if mode == 2:
                vt[:, t, :] = torch.mean(vt_k, 1)

            I = self.lnorm(torch.einsum('hv, vb->hb', self.W, v) + torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]))
            rt[:, t, :] = (I - self.theta[:, None]) / self.gamma[:, None]

        return vt[:, 1:, :], rt[:, 1:, :]

    def sample(self, v_start, chain=50, pre_gibbs_k=100, gibbs_k=20, mode=1, disable_tqdm=False, num_samples=None):
        if v_start.ndim == 1:
            num_samples = 1
        elif v_start.ndim == 2:
            num_samples = v_start.shape[1]

        if v_start is None:
            if num_samples is None:
                num_samples = 1
            v_start = torch.randint(0, 2, size=(self.n_visible, num_samples))

        #v_start = v_start.to(self.device)
        vt = torch.zeros(self.n_visible, chain + 1, num_samples, dtype=torch.float, device=self.device)
        rt = torch.zeros(self.n_hidden, chain + 1, num_samples, dtype=torch.float, device=self.device)

        vt[:, 0, :] = v_start.detach().clone().to(self.device)
        I = self.lnorm(torch.einsum('hv, vb->hb', self.W, vt[:, 0, :]))
        rt[:, 0, :] = (I - self.theta_0[:, None]) / self.gamma_0[:, None]


        std = 1 / torch.sqrt(self.gamma[:, None].repeat(1, num_samples))
        for t in tqdm(range(1, chain + 1), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                I = self.lnorm(torch.einsum('hv, vb->hb', self.W, v) + torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]))
                mean = (I - self.theta[:, None]) / self.gamma[:, None]
                h = torch.normal(mean=mean, std=std)
                v = torch.bernoulli(torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_v[:, None]))

            vt_k = torch.zeros(self.n_visible, gibbs_k, num_samples, dtype=torch.float, device=self.device)
            ht_k = torch.zeros(self.n_hidden, gibbs_k, num_samples, dtype=torch.float, device=self.device)
            for kk in range(gibbs_k):
                I = self.lnorm(torch.einsum('hv, vb->hb', self.W, v) + torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]))
                mean = (I - self.theta[:, None]) / self.gamma[:, None]
                h = torch.normal(mean=mean, std=std)
                v = torch.bernoulli(torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_v[:, None]))
                vt_k[:, kk, :] = v.detach().clone()
                ht_k[:, kk, :] = h.detach().clone()

            if mode == 1:
                vt[:, t, :] = vt_k[:, -1, :]
            if mode == 2:
                vt[:, t, :] = torch.mean(vt_k, 1)

            I = self.lnorm(torch.einsum('hv, vb->hb', self.W, v) + torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]))
            rt[:, t, :] = (I - self.theta[:, None]) / self.gamma[:, None]

        return vt[:, 1:, :], rt[:, 1:, :]



if __name__ == '__main__':
    import os, sys

    # os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from data.mock_data import create_BB
    from data.poisson_data_v import PoissonTimeShiftedData
    from data.reshape_data import reshape
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # data = create_BB(N_V=16, T=100, n_samples=20, width_vec=[4, 5, 6], velocity_vec=[1, 2])
    n_hidden = 3
    temporal_connections = torch.tensor([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ]).float()
    gaus = PoissonTimeShiftedData(
        neurons_per_population=20,
        n_populations=n_hidden, n_batches=35,
        time_steps_per_batch=1000,
        fr_mode='gaussian', delay=1, temporal_connections=temporal_connections, norm=0.36, spread_fr=[0.5, 1.5])

    gaus.plot_stats(T=100)
    plt.show()
    data = reshape(gaus.data)
    data = reshape(data, T=20, n_batches=1750)
    train, test = data[..., :1400], data[..., 1400:]
    rtrbm = RTRBM_G(train, n_hidden=3, device="cuda")
    rtrbm.learn(batch_size=50, n_epochs=5, lr_schedule='geometric_decay', max_lr=1e-4, min_lr=1e-5, start_decay=100,
                CDk=10, mom=0.6, wc=0, sp=0, x=0)

    r_data = rtrbm._parallel_recurrent_sample_r_given_v(train.to('cuda'))

    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    ax[0, 0].plot(rtrbm.errors)
    sns.heatmap(rtrbm.W.cpu(), ax=ax[0, 1])
    sns.heatmap(rtrbm.U.cpu(), ax=ax[0, 2])
    sns.heatmap(r_data[..., 0].cpu(), ax=ax[1, 0])
    sns.kdeplot(rtrbm.theta.cpu(), ax=ax[1, 1])
    sns.kdeplot(rtrbm.gamma.cpu(), ax=ax[1, 2])

    t = test.shape[1]
    vt_infer, rt_infer = rtrbm.sample(torch.tensor(test[:, 0, :]), chain=t)
    sns.heatmap(vt_infer[..., 0].cpu().detach().numpy(), ax=ax[0, 3], cbar=False)
    ax[0, 3].set_title('Infered data')
    ax[0, 3].set_xlabel('Time')
    ax[0, 3].set_ylabel('Neuron index')

    x, y = np.mean(vt_infer[:, t:, :].cpu().detach().numpy(), (1, 2)), np.mean(test[:, t:, :].detach().numpy(), (1, 2))
    mini, maxi = 0.8*np.min(np.concatenate([x.ravel(), y.ravel()])), 1.2*np.max(np.concatenate([x.ravel(), y.ravel()]))
    ax[1, 3].scatter(x, y)
    r = np.corrcoef(x, y)[0, 1]
    ax[1, 3].plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    #ax[1, 3].text(.1, .85, '$r_p={:.2f}$'.format(r), fontsize=12)
    ax[1, 3].set_title('$\langle v_i \\rangle$, $r_p={:.2f}$'.format(r))
    # ax[1, 3].set_xlim([mini, maxi])
    # ax[1, 3].set_ylim([mini, maxi])
    #print(torch.sum(torch.isnan(vt_infer)), torch.sum(torch.isinf(vt_infer)), torch.sum(torch.isnan(rt_infer)), torch.sum(torch.isinf(rt_infer)))
    plt.tight_layout()
    plt.show()


