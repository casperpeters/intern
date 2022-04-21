import torch


class RBM(torch.nn.Module):
    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None):
        super(RBM, self).__init__()
        if W is None:
            W = torch.nn.Parameter(0.01 * torch.randn(size=(n_visible, n_hidden)))
        if hbias is None:
            hbias = torch.nn.Parameter(torch.zeros(n_hidden))
        if vbias is None:
            vbias = torch.nn.Parameter(torch.zeros(n_visible))

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.input = input
        self.parameters = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        wx_b = torch.dot(v_sample, self.W) + self.hbias
        vbias_term = torch.dot(v_sample, self.vbias)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), 1)
        return -hidden_term - vbias_term

    def propdown(self, hid):
        pre_sigmoid_activation = torch.matmul(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = torch.bernoulli(v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]


class ShiftedRBM(RBM):
    def __init__(self, input, n_hidden, n_visible, U=None, W=None, hbias=None, vbias=None):
        RBM.__init__(self, input, n_visible=n_visible, n_hidden=n_hidden, W=W, hbias=hbias, vbias=vbias)

    def free_energy_given_hid_lag(self, v_sample, U, hid_lag):
        wx_b = torch.matmul(v_sample, self.W) + torch.matmul(hid_lag, U) + self.hbias
        vbias_term = torch.matmul(v_sample, self.vbias)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), 1)
        return -hidden_term - vbias_term

    def propup_given_hid_lag(self, vis, Wp, hid_lag):
        pre_sigmoid_activation = torch.matmul(vis, self.W) + torch.matmul(hid_lag, Wp) + self.hbias
        return [pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v_hid_lag(self, v0_sample, Wp, hid_lag):
        pre_activation_h1, h1_mean = self.propup_given_hid_lag(v0_sample, Wp, hid_lag)
        h1_sample = torch.bernoulli(h1_mean)
        return [pre_activation_h1, h1_mean, h1_sample]

    def gibbs_vhv_given_h_lag(self, v0, Wp, h_lag):
        pre_activation_h1, h1_mean, h1_sample = self.sample_h_given_v_hid_lag(v0, Wp, h_lag)
        pre_activation_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_activation_h1, h1_mean, h1_sample,
                pre_activation_v1, v1_mean, v1_sample]


class RTRBM(torch.nn.Module):
    def __init__(self, input, n_hidden, n_visible, time, h0=None, U=None, vbias=None, hbias=None, W=None, act=None):
        super(RTRBM, self).__init__()
        self.input = input  # n_batches, n_visibles x time
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.T = time
        if act is None:
            act = torch.sigmoid
        if h0 is None:
            h0 = torch.nn.Parameter(torch.zeros(n_hidden))
        if U is None:
            U = torch.nn.Parameter(0.01 * torch.randn(n_hidden, n_hidden))
        if W is None:
            W = torch.nn.Parameter(0.01 * torch.randn(n_visible, n_hidden))
        if vbias is None:
            vbias = torch.nn.Parameter(torch.zeros(n_visible))
        if hbias is None:
            hbias = torch.nn.Parameter(torch.zeros(n_hidden))
        self.activation = act
        self.h0 = h0
        self.U = U
        self.W = W
        self.vbias = vbias
        self.hbias = hbias
        # self.parameters = [self.h0, self.U, self.W, self.vbias, self.hbias]

        self.temporal_layers = []
        for t in range(self.T):
            self.temporal_layers += [ShiftedRBM(
                self.input[:, t*n_visible:(t+1)*n_visible],
                n_visible=n_visible,
                n_hidden=n_hidden,
                W=self.W,
                vbias=vbias,
                hbias=hbias)]

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def one_V_temporal_sampling(self, V):
        V_sample = []
        H = [self.h0]
        for t, layer in enumerate(self.temporal_layers):
            pre_sigmoid_h1, h1_mean, h1_sample, \
                pre_sigmoid_v1, v1_mean, v1_sample = \
                (layer.gibbs_vhv_given_h_lag(V[:, t * self.n_visible:(t+1) * self.n_visible], self.U, H[-1]))
            V_sample += [v1_sample]
            H += [h1_mean]
        return torch.cat(V_sample, dim=1)

    def H_given_V(self, V, U, h0):
        H = [h0]
        for t, layer in enumerate(self.temporal_layers):
            H += [layer.propup_given_hid_lag(V[:, t * self.n_visible:(t + 1) * self.n_visible], U, H[-1])[1]]
        return H[1:]

    def free_energy_RTRBM(self, V):
        H = self.H_given_V(V, self.U, self.h0)
        free_energy = []
        for t in range(self.T):
            free_energy += [self.temporal_layers[t].free_energy_given_hid_lag(V[:, t * self.n_visible:(t + 1) * self.n_visible], self.U, H[t])]
        return sum(free_energy)

    def get_cost_updates(self, lr=0.001, k=1):

        chain_start = self.input
        V_sample = self.one_V_temporal_sampling(chain_start)

        for kk in range(k-1):
            V_sample = self.one_V_temporal_sampling(V_sample)

        self.optimizer.zero_grad()

        loss = torch.mean(self.free_energy_RTRBM(self.input.detach())) - \
               torch.mean(self.free_energy_RTRBM(V_sample.detach()))
        loss.backward()
        self.optimizer.step()

        return V_sample


if __name__ == '__main__':
    import os

    os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from data.mock_data import create_BB
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data = create_BB()[:, :, :10]
    new_data = torch.zeros(data.shape[2], data.shape[0]*data.shape[1])
    for t in range(data.shape[1]):
        new_data[:, t*data.shape[0]:(t+1)*data.shape[0]] = data[:, t, :].squeeze().T

    rtrbm = RTRBM(input=new_data, n_hidden=10, n_visible=16, time=32)
    err = []
    for epoch in tqdm(range(5000)):
        V_sample = rtrbm.get_cost_updates(lr=0.001)
        # print(torch.mean((V_sample - data)**2))
        err += [torch.mean((V_sample - new_data)**2).detach().numpy()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(rtrbm.W.detach().numpy(), ax=axes[0])
    sns.heatmap(rtrbm.U.detach().numpy(), ax=axes[1])
    plt.plot(err)
    plt.show()

