'Scipt for the cRTM, visibles potential = bernoulli, hidden potential = dReLU'

import torch
from tqdm import tqdm
from scipy.special import ndtri

class RBM(object):

    def __init__(self, data, N_H=10, device=None):

        if device is None:
            self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        else:
            self.device = device

        self.dtype = torch.float
        self.data = data.float().to(self.device)
        self.N_H = N_H
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

        # HU potential, parametrization
        self.W = 0.01/self.N_V * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(self.N_V, dtype=self.dtype, device=self.device)
        self.gamma_p = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_m = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_p = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_m = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)



    def learn(self,
              n_epochs=1000,
              batchsize=10,
              CDk=10, PCD=False,
              lr=1e-3, lr_end=None, start_decay=10,
              sp=None, x=2,
              mom=0,
              wc=0,
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

        self.errors = torch.zeros(n_epochs, 1, dtype=self.dtype, device=self.device)
        self.disable = disable_tqdm

        params_C = [self.W, self.b_V, self.theta_p, self.theta_m, self.gamma_p, self.gamma_m]
        Dparams = [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in params_C]

        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0
            for batch in range(num_batches):

                # Initialize batch parameters, and compute batch input
                v_data, v_model, h_data, h_model, I_data = self.batch_parameters(batch, batchsize)
                dparams = [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in params_C]

                for i in range(0, batchsize):
                    # Contrastive Divergence
                    v_model, h_model, h_data = self.CD(v_data[:, i], CDk)

                    # Accumulate error
                    err += torch.sum((v_data[:, i] - v_model) ** 2)

                    # Compute gradients
                    dW, db_V, dtheta_p, dtheta_m, dgamma_p, dgamma_m = self.internal_grad(v_data[:, i], h_data, v_model,
                                                                                          h_model)

                    # Compute mean gradients and store values
                    dparam = [dW, db_V, dtheta_p, dtheta_m, dgamma_p, dgamma_m]

                    for j in range(len(dparam)): dparams[j] += dparam[j] / batchsize

                # Update parameters with gradient
                Dparams = self.update_params(Dparams, dparams, mom=mom, wc=wc, sp=sp, x=x)

            self.errors[epoch] = err / self.data.numel()

            if lr and lr_end and start_decay is not None:
                if start_decay > epoch:
                    lr *= r


    def batch_parameters(self, batch, batchsize):
        v_data = self.data[:, batch:batch + batchsize]
        v_model = torch.zeros_like(v_data, dtype=self.dtype, device=self.device)
        h_data = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        h_model = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        I_data = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        for i in range(batchsize):
            I_data[:, i] = torch.matmul(v_data[:, i], self.W.T)

        return v_data, v_model, h_data, h_model, I_data

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def visible_to_hidden(self, v):
        I = torch.matmul(v, self.W.T)
        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I + self.theta_m) / torch.sqrt(self.gamma_m)
        etg_plus = self.erf_times_gauss(I_plus)
        etg_min = self.erf_times_gauss(I_min)

        p_plus = torch.tensor(1 / (1 + (etg_min / torch.sqrt(self.gamma_m)) / (etg_plus / torch.sqrt(self.gamma_p))), \
                              dtype=self.dtype, device=self.device)

        sqrt2 = torch.sqrt(torch.tensor(2, dtype=self.dtype, device=self.device))
        h = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)

        for i in range(self.N_H):  # for loop is obsolete
            if torch.isnan(p_plus[i]):
                if torch.abs(I_plus[i]) > torch.abs(I_min[i]):
                    p_plus[i] = 1
                else:
                    p_plus[i] = 0
            # p_min = 1-p_plus

            is_pos = (torch.rand(1, dtype=self.dtype, device=self.device) < p_plus[i])[0]
            if is_pos:
                rmin = torch.special.erf(I_plus[i] / sqrt2)
                rmax = 1
            else:
                rmin = -1
                rmax = torch.special.erf(-I_min[i] / sqrt2)

            out = sqrt2 * torch.special.erfinv(
                rmin + (rmax - rmin) * torch.rand(1, dtype=self.dtype, device=self.device))
            if is_pos:
                h[i] = (out - I_plus[i]) / torch.sqrt(self.gamma_p[i])
            else:
                h[i] = (out + I_min[i]) / torch.sqrt(self.gamma_m[i])

            if torch.isinf(out) | torch.isnan(out) | (rmax - rmin < 1e-14):
                h[i] = 0
        return h

    def hidden_to_visible(self, h):
        return torch.bernoulli(torch.sigmoid(torch.matmul(h, self.W) + self.b_V))

    def CD(self, v_data, CDk):

        h_data = self.visible_to_hidden(v_data)
        h_model = h_data.detach().clone().to(self.device)
        for k in range(CDk):
            v_model = self.hidden_to_visible(h_model)
            h_model = self.visible_to_hidden(v_model)

        return v_model, h_model, h_data

    def internal_grad(self, v_data, h_data, v_model, h_model):
        # Data
        I_d = torch.matmul(v_data, self.W.T)
        I_plus = (-I_d + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I_d + self.theta_m) / torch.sqrt(self.gamma_m)
        phi_plus_d = self.erf_times_gauss(I_plus) / torch.sqrt(self.gamma_p)
        phi_min_d = self.erf_times_gauss(I_min) / torch.sqrt(self.gamma_m)
        p_plus_d = 1 / (1 + (phi_min_d / phi_plus_d))
        p_min_d = 1 - p_plus_d

        # Model
        I_m = torch.matmul(v_model, self.W.T)
        I_plus = (-I_m + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I_m + self.theta_m) / torch.sqrt(self.gamma_m)
        phi_plus_m = self.erf_times_gauss(I_plus) / torch.sqrt(self.gamma_p)
        phi_min_m = self.erf_times_gauss(I_min) / torch.sqrt(self.gamma_m)
        p_plus_m = 1 / (1 + (phi_min_m / phi_plus_m))
        p_min_m = 1 - p_plus_m

        # positive - negative gradient
        dtheta_p = - ((p_plus_d * ((I_d - self.theta_p) / self.gamma_p + 1 / (torch.sqrt(self.gamma_p) * phi_plus_d))) - \
                      (p_plus_m * ((I_m - self.theta_p) / self.gamma_p + 1 / (torch.sqrt(self.gamma_p) * phi_plus_m))))

        dtheta_m = - ((p_min_d * ((I_d - self.theta_m) / self.gamma_m - 1 / (torch.sqrt(self.gamma_m) * phi_min_d))) - \
                      (p_min_m * ((I_m - self.theta_m) / self.gamma_m - 1 / (torch.sqrt(self.gamma_m) * phi_min_m))))

        dgamma_p = -1 / 2 * ((p_plus_d * (1 / self.gamma_p + ((I_d - self.theta_p)
                                                              / self.gamma_p) ** 2 + (I_d - self.theta_p) / (
                                                      self.gamma_p * phi_plus_d))) -

                             (p_plus_m * (1 / self.gamma_p + ((I_m - self.theta_p)
                                                              / self.gamma_p) ** 2 + (I_m - self.theta_p) / (
                                                      self.gamma_p * phi_plus_m))))

        dgamma_m = -1 / 2 * ((p_min_d * (1 / self.gamma_m + ((I_d - self.theta_m)
                                                             / self.gamma_m) ** 2 - (I_d - self.theta_m) / (
                                                     self.gamma_m * phi_min_d))) -

                             (p_min_m * (1 / self.gamma_m + ((I_m - self.theta_m)
                                                             / self.gamma_m) ** 2 - (I_m - self.theta_m) / (
                                                     self.gamma_m * phi_min_m))))

        dW = torch.outer(h_data, v_data) - torch.outer(h_model, v_model)
        db_V = v_data - v_model

        return dW, db_V, dtheta_p, dtheta_m, dgamma_p, dgamma_m

    def update_params(self, Dparams, dparams, lr=0.01, mom=0.9, wc=0.0002, sp=None, x=2):


        [dW, db_V, dtheta_p, dtheta_m, dgamma_p, dgamma_m] = dparams
        DW, Db_V, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        Db_V = mom * Db_V + lr * db_V
        Dgamma_p = mom * Dgamma_p + lr * dgamma_p
        Dgamma_m = mom * Dgamma_m + lr * dgamma_m

        Dtheta_p = mom * Dtheta_p + lr * dtheta_p
        Dtheta_m = mom * Dtheta_m + lr * dtheta_m


        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V),
                                     [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, Db_V, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m]

        self.W += DW
        self.b_V += Db_V
        self.gamma_p += Dgamma_p
        self.theta_p += Dtheta_p
        self.gamma_m += Dgamma_m
        self.theta_m += Dtheta_m

        return Dparams

    def mean_from_inputs(self, I, I0=None):
        # First moment
        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_minus = (I + self.theta_m) / torch.sqrt(self.gamma_m)

        etg_plus = self.erf_times_gauss(I_plus)
        etg_minus = self.erf_times_gauss(I_minus)

        p_plus = 1 / (1 + (etg_minus / torch.sqrt(self.gamma_m)
                           ) / (etg_plus / torch.sqrt(self.gamma_p)))
        nans = torch.isnan(p_plus)
        p_plus[nans] = 1.0 * (torch.abs(I_plus[nans]) > torch.abs(I_minus[nans]))
        p_minus = 1 - p_plus
        mean_pos = (-I_plus + 1 / etg_plus) / torch.sqrt(self.gamma_p)
        mean_neg = (I_minus - 1 / etg_minus) / torch.sqrt(self.gamma_m)
        return mean_pos * p_plus, mean_neg * p_minus

    def mean2_from_inputs(self, I, I0=None, beta=1):
        # Second moment
        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I + self.theta_m) / torch.sqrt(self.gamma_m)

        etg_plus = self.erf_times_gauss(I_plus)
        etg_minus = self.erf_times_gauss(I_min)
        p_plus = 1 / (1 + (etg_minus / torch.sqrt(self.gamma_m)
                           ) / (etg_plus / torch.sqrt(self.gamma_p)))
        nans = torch.isnan(p_plus)
        p_plus[nans] = 1.0 * (torch.abs(I_plus[nans]) > torch.abs(I_min[nans]))
        p_minus = 1 - p_plus
        mean2_pg = 1 / self.gamma_p * (1 + I_plus ** 2 - I_plus / etg_plus)
        mean2_ng = 1 / self.gamma_m * (1 + I_min ** 2 - I_min / etg_minus)
        return p_plus * mean2_pg, p_minus * mean2_ng

    def var_from_inputs(self, I, I0=None, beta=1):
        (mu_pos, mu_neg) = self.mean_from_inputs(I)
        (mu2_pos, mu2_neg) = self.mean2_from_inputs(I)
        return (mu2_pos + mu2_neg) - (mu_pos + mu_neg) ** 2

    def cgf_dReLU(self, I):
        """"Cumulant generating function associated to the HU dReLU potential"""
        N_H, batchsize = I.shape
        cgf = torch.zeros(I.shape, dtype=self.dtype, device=self.device)

        for b in range(batchsize):
            for n in range(N_H):
                Z_plus = torch.log(self.erf_times_gauss(
                    (-I[n, b] + self.theta_p[n]) / torch.sqrt(self.gamma_p)[n]) - 0.5 * torch.log(self.gamma_p)[n])
                Z_min = torch.log(self.erf_times_gauss(
                    (I[n, b] + self.theta_m[n]) / torch.sqrt(self.gamma_m)[n]) - 0.5 * torch.log(self.gamma_m)[n])
                if Z_plus > Z_min:
                    cgf[n, b] = Z_plus + torch.log(1 + torch.exp(Z_min - Z_plus))
                else:
                    cgf[n, b] = Z_min + torch.log(1 + torch.exp(Z_plus - Z_min))
        return cgf

    def erf_times_gauss(self, x):

        a1 = torch.tensor(0.3480242, dtype=self.dtype, device=self.device)
        a2 = torch.tensor(-0.0958798, dtype=self.dtype, device=self.device)
        a3 = torch.tensor(0.7478556, dtype=self.dtype, device=self.device)
        p = torch.tensor(0.47047, dtype=self.dtype, device=self.device)

        sqrtpiover2 = torch.sqrt(torch.tensor(torch.pi / 2, dtype=self.dtype, device=self.device))
        out = torch.zeros(torch.numel(x), dtype=self.dtype, device=self.device)
        size = torch.numel(x)
        if size > 1:
            for i in range(torch.numel(x)):
                if x[i] < -6:
                    out[i] = 2 * torch.exp(x[i] ** 2 / 2)
                elif x[i] > 0:
                    t = 1 / (1 + p * x[i])
                    out[i] = t * (a1 + a2 * t + a3 * t ** 2)
                else:
                    t = 1 / (1 - p * x[i])
                    out[i] = -t * (a1 + a2 * t + a3 * t ** 2) + 2 * torch.exp(x[i] ** 2 / 2)
        elif size == 1:
            if x < -6:
                out = 2 * torch.exp(x ** 2 / 2)
            elif x > 0:
                t = 1 / (1 + p * x)
                out = t * (a1 + a2 * t + a3 * t ** 2)
            else:
                t = 1 / (1 - p * x)
                out = -t * (a1 + a2 * t + a3 * t ** 2) + 2 * torch.exp(x ** 2 / 2)

        return sqrtpiover2 * out

    def sample(self,
               v_start,
               pre_gibbs_k=100,
               gibbs_k=20,
               mode=1,
               chain=50,
               disable_tqdm=False):

        vt = torch.zeros(self.N_V, chain, dtype=self.dtype, device=self.device)
        ht = torch.zeros(self.N_H, chain, dtype=self.dtype, device=self.device)

        v = v_start

        for kk in range(pre_gibbs_k):
            h = self.visible_to_hidden(v)
            v = self.hidden_to_visible(h)

        for t in tqdm(range(chain), disable=disable_tqdm):
            vt_k = torch.zeros(self.N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = self.visible_to_hidden(v)
                v = self.hidden_to_visible(h)
                vt_k[:, kk] = v
                ht_k[:, kk] = h

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
                ht[:, t] = ht_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
                ht[:, t] = torch.mean(ht_k, 1)

        return vt, ht



if __name__=='__main__':
    from data.mock_data import create_BB
    from data.poisson_data_v import PoissonTimeShiftedData
    from data.reshape_data import reshape
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # data = create_BB(N_V=16, T=320, n_samples=10, width_vec=[4, 5, 6], velocity_vec=[1, 2])
    n_h = 3
    temporal_connections = torch.tensor([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ]).float()
    gaus = PoissonTimeShiftedData(
        neurons_per_population=20,
        n_populations=n_h, n_batches=25,
        time_steps_per_batch=1000,
        fr_mode='gaussian', delay=1, temporal_connections=temporal_connections, norm=0.36, spread_fr=[0.5, 1.5])

    gaus.plot_stats(T=100)
    plt.show()
    data = reshape(gaus.data)
    data = reshape(data, T=100, n_batches=250)
    train, test = data[..., :200], data[..., 200:]

    rbm = RBM(train, N_H=n_h, device="cpu")
    rbm.learn(batchsize=10, n_epochs=100, lr=1e-3, lr_end=9e-4, start_decay=0, mom=0, wc=0, sp=None, x=0)

    plt.plot(rbm.errors)

    # vt_infer, rt_infer = rbm.sample(test, dtype=torch.float))
    # sns.heatmap(vt_infer.cpu())
    # plt.show()
    #
    # plt.plot(rbm.errors.cpu())
    # plt.show()
    #
    # h = torch.zeros(N_H, spikes.shape[1])
    # for i in range(spikes.shape[1]):
    #     h[:, i] = rbm.visible_to_hidden(spikes[:, i].float())
    # plot_effective_coupling_VH(rbm.W, spikes.float(), h)
    # plt.show()
    # v_sampled, h_sampled = rbm.sample(data[:, 0].float(), pre_gibbs_k=100, gibbs_k=20, mode=1, chain=50)
    # plot_true_sampled(data.float(), h, v_sampled, h_sampled)
    # plt.show()
    # sns.kdeplot(np.array(h.ravel().cpu()), bw_adjust=0.1)
    # plt.show()


