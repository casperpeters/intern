import torch
import numpy as np
from tqdm import tqdm
from optim.lr_scheduler import get_lrs


class RBM(torch.nn.Module):
    def __init__(self, data=None, n_visible=784, n_hidden=500, W=None, b_h=None, b_v=None):
        super(RBM, self).__init__()
        if W is None:
            W = torch.nn.Parameter(0.01 * torch.randn(size=(n_hidden, n_visible)))
        if b_h is None:
            b_h = torch.nn.Parameter(torch.zeros(n_hidden))
        if b_v is None:
            b_v = torch.nn.Parameter(torch.zeros(n_visible))

        self.W = W
        self.b_h = b_h
        self.b_v = b_v
        self.data = data
        self.parameters = [self.W, self.b_h, self.b_v]

    def free_energy(self, v_sample):
        wx_b = torch.dot(self.W, v_sample) + self.b_h
        b_v_term = torch.dot(v_sample, self.b_v)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), 1)
        return -hidden_term - b_v_term

    def sample_h_given_v(self, v):
        h_mean = torch.sigmoid(torch.matmul(self.W, v) + self.b_h[:, None])
        h_sample = torch.bernoulli(h_mean)
        return [h_mean, h_sample]

    def sample_v_given_h(self, h):
        v_mean = torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v[:, None])
        v_sample = torch.bernoulli(v_mean)
        return [v_mean, v_sample]


class ShiftedRBM(RBM):
    def __init__(self, data, n_hidden, n_visible, U=None, W=None, b_h=None, b_v=None):
        RBM.__init__(self, data, n_visible=n_visible, n_hidden=n_hidden, W=W, b_h=b_h, b_v=b_v)

    def free_energy_given_r_lag(self, v, U, r_lag, b_init, t):
        if t == 0:
            wx_b = torch.matmul(self.W, v) + b_init[:, None]
        else:
            wx_b = torch.matmul(self.W, v) + torch.matmul(U, r_lag) + self.b_h[:, None]
        b_v_term = torch.matmul(self.b_v, v)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), 0)
        return -hidden_term - b_v_term

    def sample_h_given_v_r_lag(self, v, U, r_lag):
        r = torch.sigmoid(torch.matmul(self.W, v) + torch.matmul(U, r_lag) + self.b_h[:, None])
        return [r, torch.bernoulli(r)]

    def sample_h_given_v_b_init(self, v, b_init):
        r = torch.sigmoid(torch.matmul(self.W, v) + b_init[:, None])
        return [r, torch.bernoulli(r)]

    def CD_vhv_given_r_lag(self, v_data, U, r_lag, CDk=1):
        r_data, h_model = self.sample_h_given_v_r_lag(v_data, U, r_lag)
        for k in range(CDk - 1):
            _, v_model = self.sample_v_given_h(h_model)
            _, h_model = self.sample_h_given_v_r_lag(v_data, U, r_lag)
        _, v_model = self.sample_v_given_h(h_model)
        return [v_model, h_model, r_data]

    def CD_vhv_given_b_init(self, v_data, b_init, CDk=1):
        r_data, h_model = self.sample_h_given_v_b_init(v_data, b_init)
        for k in range(CDk - 1):
            _, v_model = self.sample_v_given_h(h_model)
            _, h_model = self.sample_h_given_v_b_init(v_data, b_init)
        _, v_model = self.sample_v_given_h(h_model)
        return [v_model, h_model, r_data]


class RTRBM(torch.nn.Module):
    def __init__(self, data, n_hidden, batch_size=1, act=None, device=None):
        super(RTRBM, self).__init__()

        if device is None:
            device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        self.device = device
        self.dtype = torch.float
        self.data = data.detach().clone().to(self.device)  # n_visible, time, n_batches
        self.n_hidden = n_hidden

        if torch.tensor(self.data.shape).shape[0] == 3:
            self.n_batches = data.shape[2] // batch_size
            self.n_visible, self.T, _ = data.shape
        else:
            self.n_batches = 1
            self.n_visible, self.T = data.shape

        if act is None:
            act = torch.sigmoid

        self.batch_size = batch_size
        self.activation = act

        self.U = torch.nn.Parameter( 0.01 * torch.randn(self.n_hidden, self.n_hidden, dtype=self.dtype, device=self.device))
        self.W = torch.nn.Parameter( 0.01 * torch.randn(self.n_hidden, self.n_visible, dtype=self.dtype, device=self.device))
        self.b_v = torch.nn.Parameter(torch.zeros(self.n_visible, dtype=self.dtype, device=self.device))
        self.b_h = torch.nn.Parameter(torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device))
        self.b_init = torch.nn.Parameter(torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device))

        self.temporal_layers = []
        for t in range(self.T):
            self.temporal_layers += [ShiftedRBM(self.data[:, t, :], n_visible=self.n_visible, n_hidden=self.n_hidden,
                                                W=self.W, b_v=self.b_v, b_h=self.b_h)]

    def contrastive_divergence(self, v, CDk):
        v_model = torch.zeros(self.n_visible, len(self.temporal_layers), self.batch_size, dtype=self.dtype,
                              device=self.device)
        r_model = torch.zeros(self.n_hidden, len(self.temporal_layers), self.batch_size, dtype=self.dtype,
                              device=self.device)
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                v_model[:, t, :], r_model[:, t, :], r_data_leg = \
                    (layer.CD_vhv_given_b_init(v[:, t, :], self.b_init, CDk=CDk))
            else:
                v_model[:, t, :], r_model[:, t, :], r_data_leg = \
                    (layer.CD_vhv_given_r_lag(v[:, t, :], self.U, r_data_leg, CDk=CDk))
        return v_model, r_model

    def sample_h_given_v_over_time(self, v, U, b_init):
        h = torch.zeros(self.n_hidden, v.shape[1], v.shape[2], dtype=self.dtype, device=self.device)
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                r_leg, h[:, t, :] = [layer.sample_h_given_v_b_init(v[:, t, :], b_init)[1]]
            else:
                r_leg, h[:, t, :] = [layer.sample_h_given_v_r_lag(v[:, t, :], U, r_leg)[1]]
        return h

    def free_energy_RTRBM(self, v):
        free_energy = 0
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                r_leg = None
                r = layer.sample_h_given_v_b_init(v[:, t, :], self.b_init)[1]
            else:
                r = layer.sample_h_given_v_r_lag(v[:, t, :], self.U, r_leg)[1]
            free_energy += torch.sum(
                self.temporal_layers[t].free_energy_given_r_lag(v[:, t, :], self.U, r_leg, self.b_init, t))
            r_leg = r.detach().clone()
        return free_energy

    def get_cost_updates(self, CDk=1):
        err, loss = 0, 0
        for batch in range(self.n_batches):
            v_model, r_model = self.contrastive_divergence(self.data[:, :, batch:batch + self.batch_size].detach(),
                                                           CDk=CDk)
            self.optimizer.zero_grad()
            loss += torch.mean(self.free_energy_RTRBM(self.data[:, :, batch:batch + self.batch_size].detach())) - \
                    torch.mean(self.free_energy_RTRBM(v_model.detach()))
            err += torch.sum(((self.data[:, :, batch:batch + self.batch_size].detach() - v_model) ** 2))
            loss.backward()
            self.optimizer.step()
        return err

    def learn(self, n_epochs, lr=1e-3, lr_schedule=None, sp=None, x=2, shuffle_batch=True, CDk=1, disable_tqdm=False,
              momentum=0, dampening=0, weight_decay=0, **kwargs):

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=momentum, dampening=dampening,
                                         weight_decay=weight_decay)

        if lr is None:
            lrs = np.array(get_lrs(mode=lr_schedule, n_epochs=n_epochs, **kwargs))
        else:
            lrs = lr * torch.ones(n_epochs)

        self.err = []
        for epoch in tqdm(range(n_epochs)):
            # Change lr with our own scheduler
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[epoch]
            self.err += [(rtrbm.get_cost_updates(CDk=CDk) / self.data.numel()).detach().clone()]

        return

    def sample(self, v_start, chain=50, pre_gibbs_k=100, gibbs_k=20, mode=1, disable_tqdm=False):

        vt = torch.zeros(self.n_visible, chain + 1, dtype=self.dtype, device=self.device)
        rt = torch.zeros(self.n_hidden, chain + 1, dtype=self.dtype, device=self.device)

        rt[:, 0] = self.activation(torch.matmul(self.W, v_start.T) + self.b_init)
        vt[:, 0] = v_start
        for t in tqdm(range(1, chain + 1), disable=disable_tqdm):
            v = vt[:, t - 1]

            # it is important to keep the burn-in inside the chain loop, because we now have time-dependency
            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(
                    self.activation(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(self.activation(torch.matmul(self.W.T, h) + self.b_v.T))

            vt_k = torch.zeros(self.n_visible, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.n_hidden, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(
                    self.activation(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(self.activation(torch.matmul(self.W.T, h) + self.b_v.T))
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
            rt[:, t] = self.activation(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))
        return vt[:, 1:], rt[:, 1:]


if __name__ == '__main__':
    import os

    # os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from data.mock_data import create_BB
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data = create_BB(N_V=16, T=320, n_samples=10, width_vec=[4, 5, 6], velocity_vec=[1, 2])

    rtrbm = RTRBM(data=data, n_hidden=8, batch_size=10, device='cpu')
    rtrbm.learn(n_epochs=100, CDk=10, lr_schedule='geometric_decay', max_lr=1e-2, min_lr=1e-3)

    # Infer from trained RTRBM and plot some results
    vt_infer, rt_infer = rtrbm.sample(torch.tensor(data[:, 160 // 2, 0]))
    _, ax = plt.subplots(2, 2, figsize=(12, 12))
    sns.heatmap(vt_infer.detach().numpy(), ax=ax[0, 0], cbar=False)
    ax[0, 0].set_title('Infered data')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Neuron index')

    ax[0, 1].plot(rtrbm.err)
    ax[0, 1].set_title('RMSE of the RTRBM over epoch')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('RMSE')

    sns.heatmap(rtrbm.W.detach().numpy(), ax=ax[1, 0])
    ax[1, 0].set_title('Visible to hidden connection')
    ax[1, 0].set_xlabel('Visible')
    ax[1, 0].set_ylabel('Hiddens')

    sns.heatmap(rtrbm.U.detach().numpy(), ax=ax[1, 1])
    ax[1, 1].set_title('Hidden to hidden connection')
    ax[1, 1].set_xlabel('Hidden(t-1)')
    ax[1, 1].set_ylabel('Hiddens(t)')
    plt.show()

