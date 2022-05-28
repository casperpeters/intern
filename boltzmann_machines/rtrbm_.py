"""
This is a practical file for testing quick ideas. Sebastian, please don't adjust this one :)
"""

import torch
import numpy as np
from tqdm import tqdm
from optim.lr_scheduler import get_lrs


class RTRBM(object):
    def __init__(self, data, n_hidden=10, device='cuda', no_bias=False, debug_mode=False):
        if not torch.cuda.is_available():
            print('cuda not available, using cpu')
            self.device = 'cpu'
        else:
            self.device = device
        self.dtype = torch.float
        self.V = data
        self.dim = self.V.ndim
        if self.dim == 2:
            self.n_visible, self.T = self.V.shape
            self.num_samples = 1
        elif self.dim == 3:
            self.n_visible, self.T, self.num_samples = self.V.shape
        else:
            raise ValueError("Data is not correctly defined: Use (n_visible, T) or (n_visible, T, num_samples) dimensions")
        self.n_hidden = n_hidden
        self.W = 0.01/self.n_visible * torch.randn(self.n_hidden, self.n_visible, dtype=self.dtype, device=self.device)
        self.U = 0.01/self.n_hidden * torch.randn(self.n_hidden, self.n_hidden, dtype=self.dtype, device=self.device)
        self.b_h = torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device)
        self.b_v = torch.zeros(self.n_visible, dtype=self.dtype, device=self.device)
        self.b_init = torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device)
        self.params = [self.W, self.U, self.b_h, self.b_v, self.b_init]
        self.no_bias = no_bias
        self.debug_mode = debug_mode
        if debug_mode:
            self.parameter_history = []

        self.Dparams = self.initialize_grad_updates()
        self.errors = []

    def learn(self, n_epochs=1000,
              lr=None,
              lr_schedule=None,
              batch_size=1,
              CDk=10,
              PCD=False,
              sp=None, x=2,
              mom=0.9, wc=0.0002,
              disable_tqdm=False,
              save_every_n_epochs=1, shuffle_batch=True,
              **kwargs):

        if self.dim == 2:
            num_batches = 1
            batch_size = 1
        elif self.dim == 3:
            num_batches = self.num_samples // batch_size
        if lr is None:
            lrs = np.array(get_lrs(mode=lr_schedule, n_epochs=n_epochs, **kwargs))
        else:
            lrs = lr * torch.ones(n_epochs)
        lr_offset = np.array(get_lrs(mode='linear_decay', n_epochs=num_batches, max_lr=1, min_lr=1e-3))
        self.Wv_mean, self.Wv_std = torch.zeros(self.n_hidden), torch.ones(self.n_hidden)
        self.Ur_mean, self.Ur_std = torch.zeros(self.n_hidden), torch.ones(self.n_hidden)

        self.disable = disable_tqdm
        self.lrs = lrs
        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0
            for batch in range(0, num_batches):
                self.dparams = self.initialize_grad_updates()

                if self.dim == 2:
                    v_data = self.V[..., None].to(self.device)
                elif self.dim == 3:
                    v_data = self.V[:, :, batch * batch_size : (batch+1) * batch_size].to(self.device)

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

        self.r = r_data

    def return_params(self):
        return [self.W, self.U, self.b_v, self.b_init, self.b_h, self.errors]

    def CD(self, v_data: torch.Tensor, r_data: torch.Tensor, CDk: int = 1, beta: float = 1.0) -> torch.Tensor:
        batchsize = v_data.shape[2]
        ht_k = torch.zeros(self.n_hidden, self.T, batchsize, CDk, dtype=self.dtype, device=self.device)
        probht_k = torch.zeros(self.n_hidden, self.T, batchsize, CDk, dtype=self.dtype, device=self.device)
        vt_k = torch.zeros(self.n_visible, self.T, batchsize, CDk, dtype=self.dtype, device=self.device)
        vt_k[..., 0] = v_data.detach().clone()
        probht_k[..., 0], ht_k[..., 0] = self._parallel_sample_r_h_given_v(v_data, r_data, beta=beta)
        for i in range(1, CDk):
            vt_k[..., i] = self._parallel_sample_v_given_h(ht_k[..., i-1], beta=beta)
            probht_k[..., i], ht_k[..., i] = self._parallel_sample_r_h_given_v(vt_k[..., i], r_data, beta=beta)
        return torch.mean(probht_k, 3), torch.mean(vt_k, 3), ht_k, vt_k

    def _parallel_sample_r_h_given_v(self, v: torch.Tensor, r: torch.Tensor, beta: float = 1.0) -> torch.Tensor:

        r0 = torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v[:, 0, :]) + self.b_init[:, None])
        r = torch.sigmoid(torch.einsum('hv, vTb->hTb', self.W, v[:, 1:, :]) +\
                          torch.einsum('hr, rTb->hTb', self.U, r[:, :-1, :]) + self.b_h[:, None, None])
        r = torch.cat([r0[:, None, :], r], 1)
        return r, torch.bernoulli(r)

    def _parallel_sample_v_given_h(self, h: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        v_prob = torch.sigmoid(beta * (torch.einsum('hv, hTb->vTb', self.W, h) + self.b_v[:, None, None]))
        return torch.bernoulli(v_prob)

    def _parallel_recurrent_sample_r_given_v(self, v: torch.Tensor) -> torch.Tensor:
        _, T, n_batch = v.shape
        r = torch.zeros(self.n_hidden, T, n_batch, device=v.device)
        r[:, 0, :] = torch.sigmoid(torch.einsum('hv, vb -> hb', self.W, v[:, 0, :]) + self.b_init[:, None])
        for t in range(1, T):
            r[:, t, :] = torch.sigmoid(torch.einsum('hv, vb -> hb', self.W, v[:, t, :]) +\
                torch.einsum('hr, rb -> hb', self.U, r[:, t - 1, :]) + self.b_h[:, None])
        return r

    def visible_to_expected_hidden(self, vt, AF=torch.sigmoid):
        T = vt.shape[1]
        rt = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        rt[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, T):
            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))
        return rt

    def visible_to_hidden(self, vt, r, AF=torch.sigmoid):
        T = vt.shape[1]
        ph_sample = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        ph_sample[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        ph_sample[:, 1:T] = AF(torch.matmul(self.W, vt[:, 1:T]).T + torch.matmul(self.U, r[:, 0:T - 1]).T + self.b_H).T
        return ph_sample, torch.bernoulli(ph_sample)

    def hidden_to_visible(self, h, AF=torch.sigmoid):
        return torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad(self, vt: torch.Tensor, rt: torch.Tensor, ht_k: torch.Tensor, vt_k: torch.Tensor, barvt: torch.Tensor, barht: torch.Tensor) -> torch.Tensor:
        Dt = torch.zeros(self.n_hidden, self.T + 1, vt.shape[2], dtype=self.dtype, device=self.device)
        for t in range(self.T - 1, -1, -1):
            Dt[:, t, :] = torch.einsum('hv, hb->vb', self.U, (Dt[:, t + 1, :] * rt[:, t, :] * (1 - rt[:, t, :]) + (rt[:, t, :] - barht[:, t, :])))

        db_init = torch.mean((rt[:, 0, :] - barht[:, 0, :]) + Dt[:, 1, :] * rt[:, 0, :] * (1 - rt[:, 0, :]), 1)
        tmp = torch.sum(Dt[:, 2:self.T, :] * (rt[:, 1:self.T - 1, :] * (1 - rt[:, 1:self.T - 1, :])), 1)
        db_H = torch.mean(torch.sum(rt[:, 1:self.T, :], 1) - torch.sum(barht[:, 1:self.T, :], 1) + tmp, 1)
        db_V = torch.mean(torch.sum(vt - barvt, 1), 1)
        dW_1 = torch.mean(torch.einsum('rTb, vTb -> rvb', Dt[:, 1:self.T, :] * rt[:, 0:self.T - 1, :] * (1 - rt[:, 0:self.T - 1, :]), vt[:, 0:self.T - 1, :]), 2)
        dW_2 = torch.mean(torch.einsum('rTb, vTb -> rvb', rt, vt) - torch.mean(torch.einsum('rTbk, vTbk -> rvbk', ht_k, vt_k), 3), 2)
        dW = dW_1 + dW_2
        dU = torch.mean(torch.einsum('rTb, hTb -> rhb', Dt[:, 2:self.T + 1, :] * (rt[:, 1:self.T, :] * (1 - rt[:, 1:self.T, :])) + rt[:, 1:self.T, :] - barht[:, 1:self.T, :], rt[:, 0:self.T - 1, :]), 2)
        return [dW, dU, db_H, db_V, db_init]

    def update_grad(self, lr=1e-3, mom=0, wc=0, x=2, sp=None):
        dW, dU, db_h, db_v, db_init = self.dparams
        DW, DU, Db_h, Db_v, Db_init = self.Dparams
        DW = mom * DW + lr * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Db_h = mom * Db_h + lr * db_h
        Db_v = mom * Db_v + lr * db_v
        Db_init = mom * Db_init + lr * db_init
        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.n_visible), [self.n_hidden, self.n_visible]) ** (x - 1) * torch.sign(self.W)
        if self.no_bias:
            Db_h, Db_v, Db_init = 0, 0, 0
        self.Dparams = [DW, DU, Db_h, Db_v, Db_init]
        for i in range(len(self.params)): self.params[i] += self.Dparams[i]
        return

    def infer(self,
              data,
              AF=torch.sigmoid,
              pre_gibbs_k=50,
              gibbs_k=10,
              mode=2,
              t_extra=0,
              disable_tqdm=False):

        T = self.T
        n_hidden = self.n_hidden
        n_visible, t1 = data.shape

        vt = torch.zeros(n_visible, T + t_extra, dtype=self.dtype, device=self.device)
        rt = torch.zeros(n_hidden, T + t_extra, dtype=self.dtype, device=self.device)
        vt[:, 0:t1] = data.float().to(self.device)

        rt[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, t1):
            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        for t in tqdm(range(t1, T + t_extra), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_v.T))

            vt_k = torch.zeros(n_visible, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(n_hidden, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_v.T))
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_v, vt_k) + torch.matmul(
                    self.b_h, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        return vt, rt


if __name__ == '__main__':
    import os

    # os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from data.mock_data import create_BB
    from data.poisson_data_v import PoissonTimeShiftedData
    from data.reshape_data import reshape
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data = create_BB(N_V=16, T=320, n_samples=10, width_vec=[4, 5, 6], velocity_vec=[1, 2])
    # n_hidden = 3
    # temporal_connections = torch.tensor([
    #     [0, -1, 1],
    #     [1, 0, -1],
    #     [-1, 1, 0]
    # ]).float()
    # gaus = PoissonTimeShiftedData(
    #     neurons_per_population=20,
    #     n_populations=n_hidden, n_batches=10,
    #     time_steps_per_batch=100,
    #     fr_mode='gaussian', delay=1, temporal_connections=temporal_connections, norm=0.36, spread_fr=[0.5, 1.5])
    #
    # gaus.plot_stats(T=100)
    # plt.show()
    # data = reshape(gaus.data)
    # data = reshape(data, T=100, n_batches=100)
    # train, test = data[..., :80], data[..., 80:]
    rtrbm = RTRBM(data, n_hidden=8, device="cpu")
    rtrbm.learn(batch_size=5, n_epochs=1000, lr=1e-3, CDk=10, mom=0.6, wc=0.0002, sp=0, x=0)

    vt_infer, rt_infer = rtrbm.infer(data[:, :280, 0], gibbs_k=1)
    #
    # k=1
    # vvt, vvs = [], []
    # for batch in range(test.shape[2]):
    #     vvt += [np.array(torch.matmul(test[:, :-k, batch], test[:, k:, batch].T) / (test.shape[1] - k)).flatten()]
    #     vvs += [np.array(torch.matmul(vt_infer[:, :-k, batch], vt_infer[:, k:, batch].T) / (vt_infer.shape[1] - k)).flatten()]
    #
    # plt.scatter(torch.mean(test, 1).ravel(), torch.mean(vt_infer, 1).ravel())
    # plt.show()
    # plt.scatter(vvt, vvs)
    # plt.show()
    # print('r2v: ' +str(np.corrcoef(np.array(torch.mean(test, 1).ravel()), np.array(torch.mean(vt_infer, 1).ravel()))[1, 0]**2))
    # print('r2v2: ' + str(np.corrcoef(vvt, vvs)[1, 0]**2))
    # sns.heatmap(vt_infer[..., 0].detach().numpy(), cbar=False)
    # plt.show()

    # Infer from trained RTRBM and plot some results
    vt_infer, rt_infer = rtrbm.infer(torch.tensor(data[:, :50, 0]), t_extra=50)

    # effective coupling
    W = rtrbm.W.detach().clone().numpy()
    U = rtrbm.U.detach().clone().numpy()
    rt = np.array(rtrbm._parallel_recurrent_sample_r_given_v(data))
    data = data.detach().numpy()
    var_h_matrix = np.reshape(np.var(rt[..., 0], 1).repeat(W.shape[1]), [W.shape[1], W.shape[0]]).T
    var_v_matrix = np.reshape(np.var(data[..., 0], 1).repeat(W.shape[0]), [W.shape[0], W.shape[1]])

    Je_Wv = np.matmul(W.T, W * var_h_matrix) / W.shape[1] ** 2
    Je_Wh = np.matmul(W * var_v_matrix, W.T) / W.shape[0] ** 2

    _, ax = plt.subplots(2, 3, figsize=(12, 12))
    sns.heatmap(vt_infer.detach().numpy(), ax=ax[0, 0], cbar=False)
    ax[0, 0].set_title('Infered data')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Neuron index')

    ax[0, 1].plot(rtrbm.errors)
    ax[0, 1].set_title('RMSE of the RTRBM over epoch')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('RMSE')

    sns.heatmap(Je_Wv, ax=ax[0, 2])
    ax[0, 2].set_title('Effective coupling V')
    ax[0, 2].set_xlabel("Visibel nodes")
    ax[0, 2].set_ylabel("Visibel nodes")

    sns.heatmap(rtrbm.W.detach().numpy(), ax=ax[1, 0])
    ax[1, 0].set_title('Visible to hidden connection')
    ax[1, 0].set_xlabel('Visible')
    ax[1, 0].set_ylabel('Hiddens')

    sns.heatmap(rtrbm.U.detach().numpy(), ax=ax[1, 1])
    ax[1, 1].set_title('Hidden to hidden connection')
    ax[1, 1].set_xlabel('Hidden(t-1)')
    ax[1, 1].set_ylabel('Hiddens(t)')

    sns.heatmap(Je_Wh, ax=ax[1, 2])
    ax[1, 2].set_title('Effective coupling H')
    ax[1, 2].set_xlabel("Hidden nodes [t]")
    ax[1, 2].set_ylabel("Hidden nodes [t]")
    plt.tight_layout()
    plt.show()