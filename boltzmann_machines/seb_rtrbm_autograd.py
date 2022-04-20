import torch


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
        if t==0:
            wx_b = torch.matmul(self.W, v) + b_init[:, None]
        else:
            wx_b = torch.matmul(self.W, v) + torch.matmul(U, r_lag) + self.b_h[:, None]
        b_v_term = torch.matmul(self.b_v, v)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), 0)
        return -hidden_term - b_v_term

    def sample_h_given_v_r_lag(self, v, U, r_lag):
        h_mean = torch.sigmoid(torch.matmul(self.W, v) + torch.matmul(U, r_lag) + self.b_h[:, None])
        h_sample = torch.bernoulli(h_mean)
        return [h_mean, h_sample]

    def gibbs_vhv_given_r_lag(self, v0, U, r_lag):
        h_mean, h_sample = self.sample_h_given_v_r_lag(v0, U, r_lag)
        v_mean, v_sample = self.sample_v_given_h(h_sample)
        return [h_mean, h_sample, v_mean, v_sample]

    def sample_h_given_v_b_init(self, v, b_init):
        h_mean = torch.sigmoid(torch.matmul(self.W, v) + b_init[:, None])
        h_sample = torch.bernoulli(h_mean)
        return [h_mean, h_sample]

    def gibbs_vhv_given_b_init(self, v0, b_init):
        h_mean = torch.sigmoid(torch.matmul(self.W, v0) + b_init[:, None])
        h_sample = torch.bernoulli(h_mean)
        v_mean, v_sample = self.sample_v_given_h(h_sample)
        return [h_mean, h_sample,
                v_mean, v_sample]


class RTRBM(torch.nn.Module):
    def __init__(self, data, n_hidden, n_visible, time, batchsize=None, b_init=None, U=None, b_v=None, b_h=None, W=None, act=None):
        super(RTRBM, self).__init__()
        self.data = data  # n_batches, n_visibles x time
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.T = time

        if batchsize is None:
            if torch.tensor(self.data.shape).shape[0] == 3:
                self.n_batches = data.shape[2]
            else:
                self.n_batches = 1
        if act is None:
            act = torch.sigmoid
        self.activation = act

        self.U = torch.nn.Parameter(0.01 * torch.randn(n_hidden, n_hidden))
        self.W = torch.nn.Parameter(0.01 * torch.randn(n_hidden, n_visible))
        self.b_v = torch.nn.Parameter(torch.zeros(n_visible))
        self.b_h = torch.nn.Parameter(torch.zeros(n_hidden))
        self.b_init = torch.nn.Parameter(torch.zeros(n_hidden))

        # self.parameters = [self.b_init, self.U, self.W, self.b_v, self.b_h]

        self.temporal_layers = []
        for t in range(self.T):
            self.temporal_layers += [ShiftedRBM(
                self.data[:, t, :],
                n_visible=n_visible,
                n_hidden=n_hidden,
                W=self.W,
                b_v=b_v,
                b_h=b_h)]

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def one_V_temporal_sampling(self, V, batchsize=None):
        V_sample = torch.zeros(self.n_visible, len(self.temporal_layers), self.n_batches)
        R = []
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                h_mean, h_sample, v_mean, v_sample = \
                    (layer.gibbs_vhv_given_b_init(V[:, t, :], self.b_init))
            else:
                h_mean, h_sample, v_mean, v_sample = \
                    (layer.gibbs_vhv_given_r_lag(V[:, t, :], self.U, R[-1]))

            V_sample[:, t, :] = v_sample.detach().clone()
            R += [h_mean]
        return V_sample

    def sample_h_given_v_over_time(self, V, U, b_init):
        H = []
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                H += [layer.sample_h_given_v_b_init(V[:, t, :], b_init)[1]]
            else:
                H += [layer.sample_h_given_v_r_lag(V[:, t, :], U, H[-1])[1]]
        return H

    def free_energy_RTRBM(self, V):
        H = self.sample_h_given_v_over_time(V, self.U, self.b_init)
        free_energy = []
        for t in range(self.T):
                free_energy += [self.temporal_layers[t].free_energy_given_r_lag(V[:, t, :], self.U, H[t], self.b_init, t)]
        return sum(free_energy)

    def get_cost_updates(self, lr=0.001, k=1):

        chain_start = self.data
        V_sample = self.one_V_temporal_sampling(chain_start)

        for kk in range(k-1):
            V_sample = self.one_V_temporal_sampling(V_sample)

        self.optimizer.zero_grad()

        loss = torch.mean(self.free_energy_RTRBM(self.data.detach())) - \
               torch.mean(self.free_energy_RTRBM(V_sample.detach()))
        loss.backward()
        self.optimizer.step()

        return V_sample


if __name__ == '__main__':
    import os

    #os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from data.mock_data import create_BB
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    new_data = create_BB(T=320)[:, :, :10]# N_V=16, T=32, n_batch=10, N_H=8
    # new_data = torch.zeros(data.shape[2], data.shape[0]*data.shape[1])
    # for t in range(data.shape[1]):
    #     new_data[:, t*data.shape[0]:(t+1)*data.shape[0]] = data[:, t, :].squeeze().T

    rtrbm = RTRBM(data=new_data, n_hidden=8, n_visible=16, time=320)
    err = []
    for epoch in tqdm(range(1000)):
        V_sample = rtrbm.get_cost_updates(lr=0.001)
        # print(torch.mean((V_sample - data)**2))
        err += [torch.mean((V_sample - new_data)**2).detach().numpy()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(rtrbm.W.detach().numpy(), ax=axes[0])
    sns.heatmap(rtrbm.U.detach().numpy(), ax=axes[1])
    plt.plot(err)
    plt.show()

