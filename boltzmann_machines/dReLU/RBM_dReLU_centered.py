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

        # Mean input
        self.mu_I = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)

        # contraints
        self.gamma_min = torch.tensor(0.05, dtype=self.dtype, device=self.device)
        self.jump_max = torch.tensor(20, dtype=self.dtype, device=self.device)
        self.gamma_drop_max = torch.tensor(0.75, dtype=self.dtype, device=self.device)

        # Variables
        self.W = 0.01/self.N_V * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(self.N_V, dtype=self.dtype, device=self.device)
        self.gamma_p = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_m = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_p = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_m = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)

        # Change of variables for batch normalization
        self.gamma = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.theta = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.delta = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.eta = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)

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

                # Update batch normalization
                self.update_batch_norm(mu_I=torch.mean(I_data, 1), I=I_data, batch=batch, num_batches=num_batches)  # lr EMA

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

                # Extract mean gradients of the batch
                [dW, db_V, dtheta_p, dtheta_m, dgamma_p, dgamma_m] = dparams

                # Change to centered gradients
                dgamma, dtheta, ddelta, deta = self.centered_grad(dgamma_p, dgamma_m, dtheta_p, dtheta_m)

                if torch.sum(torch.abs(dgamma)) > 100:
                    a = 1
                if torch.sum(torch.abs(dtheta)) > 100:
                    a = 1
                if torch.sum(torch.abs(ddelta)) > 100:
                    a = 1
                if torch.sum(torch.abs(deta)) > 100:
                    a = 1

                # Update centered gradients with cross-derivatives
                dW, dgamma, dtheta, ddelta, deta = self.update_grad_batch_norm(v_data, I_data, dW, dgamma, dtheta,
                                                                               ddelta, deta)
                # Update centered parameters with gradient
                Dparams = self.update_params(Dparams, dW, db_V, dgamma, dtheta, ddelta, deta, mom=mom, wc=wc, sp=sp,
                                             x=x)
                # Recompute original parameters of HU dReLU potential
                self.recompute_params()


            self.errors[epoch] = err / self.data.numel()
            #print(self.errors[epoch])

            if lr and lr_end and start_decay is not None:
                if start_decay >= epoch:
                    lr *= r

    def batch_parameters(self, batch, batchsize):
        v_data = self.data[:, batch:batch + batchsize].detach().clone().to(self.device)
        v_model = torch.zeros_like(v_data, dtype=self.dtype, device=self.device)
        h_data = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        h_model = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        I_data = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        for i in range(batchsize):
            I_data[:, i] = torch.matmul(v_data[:, i], self.W.T)

        return v_data, v_model, h_data, h_model, I_data

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
                rmax = torch.tensor(1, dtype=self.dtype, device=self.device)
            else:
                rmin = torch.tensor(-1, dtype=self.dtype, device=self.device)
                rmax = torch.special.erf(-I_min[i] / sqrt2)

            out = sqrt2 * torch.special.erfinv(rmin + (rmax - rmin) * torch.rand(1, dtype=self.dtype, device=self.device))

            if is_pos:
                h[i] = (out - I_plus[i]) / torch.sqrt(self.gamma_p[i])
            else:
                h[i] = (out + I_min[i]) / torch.sqrt(self.gamma_m[i])

            if torch.isinf(out) | torch.isnan(out) | (rmax - rmin < 1e-14):
                h[i] = 0

        if torch.sum(torch.isnan(h))>0:
            a=1

        return h

    def hidden_to_visible(self, h):
        return torch.bernoulli(torch.sigmoid(torch.matmul(h, self.W) + self.b_V))

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

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
        nans = torch.isnan(p_plus_d)
        p_plus_d[nans] = 1.0 * (torch.abs(I_plus[nans]) > torch.abs(I_min[nans]))
        p_min_d = 1 - p_plus_d

        # Model
        I_m = torch.matmul(v_model, self.W.T)
        I_plus = (-I_m + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I_m + self.theta_m) / torch.sqrt(self.gamma_m)
        phi_plus_m = self.erf_times_gauss(I_plus) / torch.sqrt(self.gamma_p)
        phi_min_m = self.erf_times_gauss(I_min) / torch.sqrt(self.gamma_m)
        p_plus_m = 1 / (1 + (phi_min_m / phi_plus_m))
        nans = torch.isnan(p_plus_m)
        p_plus_m[nans] = 1.0 * (torch.abs(I_plus[nans]) > torch.abs(I_min[nans]))
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

        if torch.sum(torch.isnan(dgamma_m))>0:
            a=1
        if torch.sum(torch.isnan(dgamma_p))>0:
            b=1
        if torch.sum(torch.isnan(dtheta_m))>0:
            c=1
        if torch.sum(torch.isnan(dtheta_p))>0:
            d=1
        if torch.sum(torch.isnan(dW))>0:
            e=1
        if torch.sum(torch.isnan(db_V))>0:
            f=1

        return dW, db_V, dtheta_p, dtheta_m, dgamma_p, dgamma_m

    def centered_grad(self, dgamma_p, dgamma_m, dtheta_p, dtheta_m):
        """"The change of variables in the gradient to the centered cRBM with dReLU potential"""
        dgamma = dgamma_p / (1 + self.eta) - dgamma_m / (1 - self.eta)
        dtheta = - dtheta_p - dtheta_m
        ddelta = dtheta_p / (1 + self.eta) - dtheta_m / (1 - self.eta)
        deta = - self.gamma * (dgamma_p / (1 + self.eta) ** 2 - dgamma_m / (1 - self.eta) ** 2) \
               - self.delta * (dtheta_p / (1 + self.eta) ** 2 - dtheta_m / (1 + self.eta) ** 2)

        if torch.sum(torch.isnan(dgamma))>0:
            a=1
        if torch.sum(torch.isnan(dtheta))>0:
            b=1
        if torch.sum(torch.isnan(ddelta))>0:
            c=1
        if torch.sum(torch.isnan(deta))>0:
            d=1
        return [dgamma, dtheta, ddelta, deta]

    def update_grad_batch_norm(self, v_data, I, dW, dgamma, dtheta, ddelta, deta):

        dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_deta, dgamma_dw = self.get_cross_derivatives_dReLU(
            v_data, I, self.gamma, self.theta, self.eta, self.delta)

        dtheta += dgamma * dgamma_dtheta
        ddelta += dgamma * dgamma_ddelta
        deta += dgamma * dgamma_deta

        dW += torch.outer(dtheta, dtheta_dw)
        dW += torch.reshape(dgamma.repeat(self.N_V), [self.N_H, self.N_V]) * dgamma_dw

        if torch.sum(torch.isnan(dtheta))>0:
            a=1
        if torch.sum(torch.isnan(ddelta))>0:
            b=1
        if torch.sum(torch.isnan(deta))>0:
            c=1
        if torch.sum(torch.isnan(dW))>0:
            d=1

        return dW, dgamma, dtheta, ddelta, deta

    def update_batch_norm(self, mu_I, I, batch, num_batches):
        batchsize = I.shape[1]
        dmu_I = mu_I - self.mu_I
        self.mu_I = mu_I
        self.theta += dmu_I
        self.theta_p += dmu_I
        self.theta_m += dmu_I

        if torch.sum(torch.isnan(self.gamma))>0:
            a=1

        expectation = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)
        variance = torch.zeros(self.N_H, batchsize, dtype=self.dtype, device=self.device)

        for i in range(batchsize):
            positive_mean, negative_mean = self.mean_h_given_I(I[:, i])
            expectation[:, i] = (positive_mean + negative_mean) * self.gamma
            variance[:, i] = (self.var_from_inputs(I[:, i]) * self.gamma - 1)

        var_e = torch.mean(expectation ** 2, 1) - torch.mean(expectation, 1) ** 2
        mean_v = torch.mean(variance, 1)
        new_gamma = (1 + mean_v + torch.sqrt((1 + mean_v) ** 2 + 4 * var_e)) / 2

        # implement gamma constraints
        gamma_min = torch.max(torch.max(
            self.gamma_min,  # gamma cannot be too small
            self.gamma_drop_max * self.gamma),  # cannot drop too quickly.
            # The jump cannot be too large.
            torch.max(-self.delta, torch.tensor(0)) * (torch.sqrt(1 - self.eta) + \
                                                       torch.sqrt(1 + self.eta)) / self.jump_max)

        lr = 0.1 * torch.logspace(1, -2, num_batches, base=10,  dtype=self.dtype, device=self.device)[batch]
        self.gamma = torch.max((1 - lr) * self.gamma + lr * new_gamma, gamma_min)

        self.gamma_p = self.gamma / (1 + self.eta)
        self.gamma_m = self.gamma / (1 - self.eta)

        if torch.sum(torch.isnan(self.gamma))>0:
            a=1
        if torch.sum(torch.isnan(self.gamma_p))>0:
            b=1
        if torch.sum(torch.isnan(self.gamma_m))>0:
            c=1

        return

    def update_params(self, Dparams, dW, db_V, dgamma, dtheta, ddelta, deta, lr=0.01, mom=0.9, wc=0.0002, sp=None, x=2):

        DW, Db_V, Dgamma, Dtheta, Ddelta, Deta = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        Db_V = mom * Db_V + lr * db_V
        Dgamma = mom * Dgamma + lr * dgamma
        Dtheta = mom * Dtheta + lr * dtheta
        Ddelta = mom * Ddelta + lr * ddelta
        Deta = mom * Deta + lr * deta

        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V),
                                     [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, Db_V, Dgamma, Dtheta, Ddelta, Deta]

        self.W += DW
        self.b_V += Db_V
        self.gamma += Dgamma
        self.theta += Dtheta
        self.delta += Ddelta
        self.eta += Deta

        # constraints
        self.eta = torch.max(self.eta, torch.tensor(-0.95, dtype=self.dtype, device=self.device))
        self.eta = torch.min(self.eta, torch.tensor(0.95, dtype=self.dtype, device=self.device))
        self.gamma = torch.max(self.gamma_min, self.gamma)

        if torch.sum(torch.isnan(self.W))>0:
            a=1
        if torch.sum(torch.isnan(self.b_V))>0:
            b=1
        if torch.sum(torch.isnan(self.gamma))>0:
            c=1
        if torch.sum(torch.isnan(self.theta))>0:
            d=1
        if torch.sum(torch.isnan(self.delta))>0:
            e=1
        if torch.sum(torch.isnan(self.eta))>0:
            f=1
        return Dparams

    def recompute_params(self):
        # Recompute parameters
        self.gamma_p = self.gamma / (1 + self.eta)
        self.gamma_m = self.gamma / (1 - self.eta)
        self.theta_p = self.theta + self.delta / torch.sqrt(1 + self.eta)
        self.theta_m = self.theta - self.delta / torch.sqrt(1 - self.eta)

    def mean_h_given_I(self, I, I0=None):
        # First moment
        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I + self.theta_m) / torch.sqrt(self.gamma_m)

        etg_plus = self.erf_times_gauss(I_plus)
        etg_min = self.erf_times_gauss(I_min)

        p_plus = 1 / (1 + (etg_min / torch.sqrt(self.gamma_m)
                           ) / (etg_plus / torch.sqrt(self.gamma_p)))
        nans = torch.isnan(p_plus)

        p_plus[nans] = 1.0 * (torch.abs(I_plus[nans]) > torch.abs(I_min[nans]))

        p_min = 1 - p_plus
        mean_pos = (-I_plus + 1 / etg_plus) / torch.sqrt(self.gamma_p)
        mean_neg = (I_min - 1 / etg_min) / torch.sqrt(self.gamma_m)

        if torch.sum(torch.isnan(mean_pos))>0:
            a=1
        if torch.sum(torch.isnan(mean_neg))>0:
            b=1

        return mean_pos * p_plus, mean_neg * p_min

    def mean_h2_given_I(self, I, I0=None, beta=1):
        # Second moment
        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I + self.theta_m) / torch.sqrt(self.gamma_m)

        etg_plus = self.erf_times_gauss(I_plus)
        etg_min = self.erf_times_gauss(I_min)
        p_plus = 1 / (1 + (etg_min / torch.sqrt(self.gamma_m)
                           ) / (etg_plus / torch.sqrt(self.gamma_p)))
        nans = torch.isnan(p_plus)
        p_plus[nans] = 1.0 * (torch.abs(I_plus[nans]) > torch.abs(I_min[nans]))
        p_min = 1 - p_plus
        mean2_pg = 1 / self.gamma_p * (1 + I_plus ** 2 - I_plus / etg_plus)
        mean2_ng = 1 / self.gamma_m * (1 + I_min ** 2 - I_min / etg_min)

        if torch.sum(torch.isnan(mean2_pg))>0:
            a=1
        if torch.sum(torch.isnan(mean2_ng))>0:
            b=1

        return p_plus * mean2_pg, p_min * mean2_ng

    def var_from_inputs(self, I, I0=None, beta=1):
        (mu_pos, mu_neg) = self.mean_h_given_I(I)
        (mu2_pos, mu2_neg) = self.mean_h2_given_I(I)
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

        sqrtpiover2 = torch.sqrt(torch.tensor(torch.pi/2, dtype=self.dtype, device=self.device))
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
                out= 2 * torch.exp(x ** 2 / 2)
            elif x > 0:
                t = 1 / (1 + p * x)
                out = t * (a1 + a2 * t + a3 * t ** 2)
            else:
                t = 1 / (1 - p * x)
                out = -t * (a1 + a2 * t + a3 * t ** 2) + 2 * torch.exp(x** 2 / 2)

        return sqrtpiover2 * out

    def get_cross_derivatives_dReLU(self, V, I, gamma, theta, eta, delta, weights=None):
        I = I.T
        V = V.T
        M = I.shape[1]  # Number of hiddens
        N = V.shape[1]  # Number of visibles
        B = I.shape[0]  # Batchsize

        if weights == None:
            weights = torch.ones(B, dtype=self.dtype, device=self.device)

        sum_weights = torch.sum(weights)
        mean_V = torch.matmul(weights, V) / sum_weights

        mean_e = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_e2 = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_v = torch.zeros(M, dtype=self.dtype, device=self.device)
        dmean_v_dgamma = torch.zeros(M, dtype=self.dtype, device=self.device)
        dmean_v_dtheta = torch.zeros(M, dtype=self.dtype, device=self.device)
        dmean_v_ddelta = torch.zeros(M, dtype=self.dtype, device=self.device)
        dmean_v_deta = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_eXde_dgamma = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_eXde_dtheta = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_eXde_ddelta = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_eXde_deta = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_de_dgamma = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_de_dtheta = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_de_ddelta = torch.zeros(M, dtype=self.dtype, device=self.device)
        mean_de_deta = torch.zeros(M, dtype=self.dtype, device=self.device)
        s1 = torch.empty((B, M), dtype=self.dtype, device=self.device)
        s2 = torch.empty((B, M), dtype=self.dtype, device=self.device)
        s3 = torch.empty((B, M), dtype=self.dtype, device=self.device)

        for m in range(M):
            gamma_ = gamma[m]
            theta_ = theta[m]
            delta_ = delta[m]
            eta_ = eta[m]
            dI_plus_dI = -torch.sqrt((1 + eta_) / gamma_)
            dI_min_dI = torch.sqrt((1 - eta_) / gamma_)
            dI_plus_ddelta = 1 / torch.sqrt(gamma_*(1+eta_))
            dI_min_ddelta = 1 / torch.sqrt(gamma_*(1-eta_))
            d2I_plus_dgammadI = torch.sqrt((1 + eta_) / gamma_ ** 3) / 2
            d2I_plus_ddeltadI = torch.tensor(0, dtype=self.dtype, device=self.device)
            d2I_plus_detadI = -1 / (2 * torch.sqrt((1 + eta_) * gamma_))
            d2I_min_dgammadI = -torch.sqrt((1 - eta_) / gamma_ ** 3) / 2
            d2I_min_ddeltadI = torch.tensor(0, dtype=self.dtype, device=self.device)
            d2I_min_detadI = -1 / (2 * torch.sqrt((1 - eta_) * gamma_))

            for b in range(B):
                I_ = I[b, m]
                weights_ = weights[b]
                I_plus = delta_/torch.sqrt((1+eta_)*gamma_) - torch.sqrt(1 + eta_) * (I_ - theta_) / torch.sqrt(gamma_)
                I_min = delta_/torch.sqrt((1-eta_)*gamma_) - torch.sqrt(1 - eta_) * (I_ - theta_) / torch.sqrt(gamma_)
                etg_plus = self.erf_times_gauss(I_plus)
                etg_min = self.erf_times_gauss(I_min)

                Z = etg_plus * torch.sqrt(1 + eta_) + etg_min * torch.sqrt(1 - eta_)

                p_plus = etg_plus*torch.sqrt(1+eta_) / Z

                if torch.isnan(p_plus):
                    if torch.abs(I_plus) > torch.abs(I_min):
                        p_plus = 1
                    else:
                        p_plus = 0
                p_min = 1 - p_plus

                e = (I_ - theta_) * (1 + eta_ * (p_plus - p_min)) - delta_*(p_plus-p_min) + 2*eta_*torch.sqrt(gamma_)/Z

                v = eta_ * (p_plus - p_min) + p_plus * p_min * (2*delta_/torch.sqrt(gamma_) - 2*eta_*(I_-theta_)/torch.sqrt(gamma_)) * \
                    2*delta_/torch.sqrt(gamma_)-2*eta_*(I_-theta_)/torch.sqrt(gamma_) - torch.sqrt(1+eta_)/etg_plus - torch.sqrt(1-eta_)/etg_min

                dI_plus_dgamma = -1 / (2 * gamma_) * I_plus
                dI_min_dgamma = -1 / (2 * gamma_) * I_min
                dI_plus_deta = -1.0 / \
                               (2 * torch.sqrt(gamma_ * (1 + eta_))) * (I_ - theta_ + delta_/(1+eta_))
                dI_min_deta = -1.0 / \
                                (2 * torch.sqrt(gamma_ * (1 - eta_))) * (I_ - theta_- delta_/(1-eta_))

                dp_plus_dI = p_plus * p_min * \
                             ((I_plus - 1 / etg_plus) * dI_plus_dI -
                              (I_min - 1 / etg_min) * dI_min_dI)

                dp_plus_ddelta = p_plus * p_min * \
                                 ((I_plus - 1 / etg_plus) * dI_plus_ddelta -
                                  (I_min - 1 / etg_min) * dI_min_ddelta)

                dp_plus_dgamma = p_plus * p_min * \
                                 ((I_plus - 1 / etg_plus) * dI_plus_dgamma -
                                  (I_min - 1 / etg_min) * dI_min_dgamma)

                dp_plus_deta = p_plus * p_min * \
                                 ((I_plus - 1 / etg_plus) * dI_plus_deta -
                                  (I_min - 1 / etg_min) * dI_min_deta)

                d2p_plus_dI2 = -(p_plus - p_min) * p_plus * p_min * (
                        (I_plus - 1 / etg_plus) * dI_plus_dI - (I_min - 1 / etg_min) * dI_min_dI) ** 2 \
                               + p_plus * p_min * ((dI_plus_dI) ** 2 * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (
                    dI_min_dI) ** 2 * (1 + (I_min - 1 / etg_min) / etg_min))

                d2p_plus_dgammadI = -(p_plus - p_min) * p_plus * p_min * ((
                        (I_plus - 1 / etg_plus) * dI_plus_dgamma - (I_min - 1 / etg_min) * dI_min_dI) * \
                        (I_plus - 1 / etg_plus) * dI_plus_dI - (I_min - 1 / etg_min) * dI_min_dgamma) + \
                        p_plus * p_min * ((I_plus - 1 / etg_plus) * d2I_plus_dgammadI - (I_min - 1 / etg_min) * d2I_min_dgammadI + \
                                            (dI_plus_dgamma * dI_plus_dI * (1 + (I_plus - 1 / etg_plus) / etg_plus) -
                                            (dI_min_dgamma * dI_min_dI * (1 + (I_min - 1 / etg_min) / etg_min))))


                d2p_plus_ddeltadI = -(p_plus - p_min) * p_plus * p_min * ((
                        (I_plus - 1 / etg_plus) * dI_plus_ddelta - (I_min - 1 / etg_min) * dI_min_dI) * \
                        (I_plus - 1 / etg_plus) * dI_plus_dI - (I_min - 1 / etg_min) * dI_min_ddelta) + \
                        p_plus * p_min * ((I_plus - 1 / etg_plus) * d2I_plus_ddeltadI - (I_min - 1 / etg_min) * d2I_min_ddeltadI + \
                                            (dI_plus_ddelta * dI_plus_dI * (1 + (I_plus - 1 / etg_plus) / etg_plus) -
                                            (dI_min_ddelta * dI_min_dI * (1 + (I_min - 1 / etg_min) / etg_min))))

                d2p_plus_detadI = -(p_plus - p_min) * p_plus * p_min * ((
                        (I_plus - 1 / etg_plus) * dI_plus_deta - (I_min - 1 / etg_min) * dI_min_dI) * \
                        (I_plus - 1 / etg_plus) * dI_plus_dI - (I_min - 1 / etg_min) * dI_min_deta) + \
                        p_plus * p_min * ((I_plus - 1 / etg_plus) * d2I_plus_detadI - (I_min - 1 / etg_min) * d2I_min_detadI + \
                                            (dI_plus_deta * dI_plus_dI * (1 + (I_plus - 1 / etg_plus) / etg_plus) -
                                            (dI_min_deta * dI_min_dI * (1 + (I_min - 1 / etg_min) / etg_min))))

                dlogZ_dI = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_dI +
                            p_min * (I_min - 1 / etg_min) * dI_min_dI)

                dlogZ_ddelta = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_ddelta +
                                p_min * (I_min - 1 / etg_min) * dI_min_ddelta)

                dlogZ_dgamma = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_dgamma +
                                p_min * (I_min - 1 / etg_min) * dI_min_dgamma)

                dlogZ_deta = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_deta +
                                p_min * (I_min - 1 / etg_min) * dI_min_deta)

                de_dI = (1 + v)
                de_dtheta = -de_dI

                de_dgamma = 2 * ((I_ - theta_) * eta_ - delta_) * dp_plus_dgamma + eta_/(Z*torch.sqrt(gamma_))\
                            - 2*eta_*torch.sqrt(gamma_) / Z * dlogZ_dgamma

                de_ddelta = -(p_plus -p_min) + 2 * ((I_ - theta_) * eta_ - delta_) * dp_plus_ddelta - 2*eta_*torch.sqrt(gamma_) / Z

                de_deta = (I_ - theta_) * (p_plus - p_min) + 2 * ((I_ - theta_) * eta_ - delta_) * dp_plus_deta \
                        + 2*torch.sqrt(gamma_) / Z - 2*eta_*torch.sqrt(gamma_) / Z * dlogZ_deta

                dv_dI = 4 * eta_ * dp_plus_dI \
                        + (2 * (I_ - theta_) * eta_ - delta_) * d2p_plus_dI2 \
                        - 2 * eta_ / (torch.sqrt(gamma_) * Z) * (de_dI - e * dlogZ_dI)

                dv_dtheta = -dv_dI

                dv_dgamma = 2 * eta_ * dp_plus_dgamma \
                            + (2 * (I_ - theta_) * eta_ - delta_) * d2p_plus_dgammadI \
                            + 2 * eta_ / (Z * torch.sqrt(gamma_)) * \
                            (e / (2 * gamma_) + e * dlogZ_dgamma - de_dgamma)

                dv_ddelta = 2 * eta_ * dp_plus_ddelta - 2 * dp_plus_dI \
                            + (2 * (I_ - theta_) * eta_ - delta_) * d2p_plus_ddeltadI \
                            + 2 * eta_ / (Z * torch.sqrt(gamma_)) * \
                            (e * dlogZ_ddelta - de_ddelta)

                dv_deta = (p_plus - p_min) \
                          + 2 * eta_ * dp_plus_deta \
                          + 2 * (I_ - theta_) * dp_plus_dI \
                          + (2 * (I_ - theta_) * eta_ - delta_) * d2p_plus_detadI \
                          - 2 * 1 / (Z * torch.sqrt(gamma_)) * \
                          (e - eta_ * e * dlogZ_deta + eta_ * de_deta)


                mean_e[m] += e * weights_
                mean_e2[m] += e ** 2 * weights_
                mean_v[m] += v * weights_
                mean_de_dgamma[m] += de_dgamma * weights_
                mean_de_dtheta[m] += de_dtheta * weights_
                mean_de_ddelta[m] += de_ddelta * weights_
                mean_de_deta[m] += de_deta * weights_
                mean_eXde_dgamma[m] += e * de_dgamma * weights_
                mean_eXde_dtheta[m] += e * de_dtheta * weights_
                mean_eXde_ddelta[m] += e * de_ddelta * weights_
                mean_eXde_deta[m] += e * de_deta * weights_
                dmean_v_dgamma[m] += dv_dgamma * weights_
                dmean_v_dtheta[m] += dv_dtheta * weights_
                dmean_v_ddelta[m] += dv_ddelta * weights_
                dmean_v_deta[m] += dv_deta * weights_

                s1[b, m] = (dv_dI * weights_)
                s2[b, m] = (e * de_dI * weights_)
                s3[b, m] = (de_dI * weights_)

        dmean_v_dw = torch.matmul(s1.T, V)
        dvar_e_dw = torch.matmul(s2.T, V)
        tmp3 = torch.matmul(s3.T, V)

        mean_e /= sum_weights
        mean_e2 /= sum_weights
        mean_v /= sum_weights
        mean_de_dgamma /= sum_weights
        mean_de_dtheta /= sum_weights
        mean_de_ddelta /= sum_weights
        mean_de_deta /= sum_weights
        mean_eXde_dgamma /= sum_weights
        mean_eXde_dtheta /= sum_weights
        mean_eXde_ddelta /= sum_weights
        mean_eXde_deta /= sum_weights
        dmean_v_dgamma /= sum_weights
        dmean_v_dtheta /= sum_weights
        dmean_v_ddelta /= sum_weights
        dmean_v_deta /= sum_weights
        dmean_v_dw /= sum_weights
        dvar_e_dw /= sum_weights
        tmp3 /= sum_weights

        var_e = mean_e2 - mean_e ** 2
        dvar_e_dgamma = 2 * (mean_eXde_dgamma - mean_e * mean_de_dgamma)
        dvar_e_dtheta = 2 * (mean_eXde_dtheta - mean_e * mean_de_dtheta)
        dvar_e_ddelta = 2 * (mean_eXde_ddelta - mean_e * mean_de_ddelta)
        dvar_e_deta = 2 * (mean_eXde_deta - mean_e * mean_de_deta)
        dtheta_dw = mean_V

        tmp = torch.sqrt((1 + mean_v) ** 2 + 4 * var_e)
        denominator = (tmp - dvar_e_dgamma -
                       dmean_v_dgamma * (1 + mean_v + tmp) / 2)
        dgamma_dtheta = (dvar_e_dtheta + dmean_v_dtheta *
                         (1 + mean_v + tmp) / 2) / denominator
        dgamma_ddelta = (dvar_e_ddelta + dmean_v_ddelta *
                         (1 + mean_v + tmp) / 2) / denominator
        dgamma_deta = (dvar_e_deta + dmean_v_deta *
                       (1 + mean_v + tmp) / 2) / denominator

        for m in range(M):
            for n in range(N):
                dvar_e_dw[m, n] -= mean_e[m] * tmp3[m, n]
        dvar_e_dw *= 2

        dgamma_dw = torch.zeros((M, N), dtype=self.dtype, device=self.device)
        for m in range(M):
            for n in range(N):
                dgamma_dw[m, n] = (dvar_e_dw[m, n] + dmean_v_dw[m, n] /
                                   2 * (1 + mean_v[m] + tmp[m])) / denominator[m]


        if torch.sum(torch.isnan(dtheta_dw))>0:
            a=1
        if torch.sum(torch.isnan(dgamma_dtheta))>0:
            b=1
        if torch.sum(torch.isnan(dgamma_ddelta))>0:
            c=1
        if torch.sum(torch.isnan(dgamma_deta))>0:
            d=1
        if torch.sum(torch.isnan(dgamma_dw))>0:
            e=1

        return dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_deta, dgamma_dw

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


# This is an example run:


import sys
sys.path.append(r'D:\RU\OneDrive\Intern\rtrbm_master')

from utils.plots import *
from data.mock_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


N_H = 4
spikes, coordinates, pop_idx = create_complex_artificial_data(n_neurons=80,
                                   t_max=250,
                                   n_populations=4,
                                   mean_firing_rate=0.2,
                                   population_correlations=[0.1, 0.6, 0.6, 0.8],
                                   neuron_population_correlation=[0.6, 0.9, 0.9, 0.9],
                                   time_shifts=None,
                                   permute=True)

spikes = torch.tensor(spikes)
sns.heatmap(spikes)
plt.show()

rbm = RBM(spikes, N_H=N_H, device="cpu")

rbm.learn(batchsize=20, n_epochs=100, lr=5e-4, lr_end=None, start_decay=None, mom=0.9, wc=0, sp=0.02, x=2, disable_tqdm=False)

#vt_infer, rt_infer = rbm.sample(torch.tensor(voxel_spike[:, 1], dtype=torch.float))
#sns.heatmap(vt_infer.cpu())
#plt.show()

plt.plot(rbm.errors.cpu())
plt.show()

h = torch.zeros(N_H, spikes.shape[1])
for i in range(spikes.shape[1]):
    h[:, i] = rbm.visible_to_hidden(spikes[:, i].float())
plot_effective_coupling_VH(rbm.W, spikes.float(), h)
plt.show()
#v_sampled, h_sampled = rbm.sample(data[:, 0].float(), pre_gibbs_k=100, gibbs_k=20, mode=1, chain=50)
#plot_true_sampled(data.float(), h, v_sampled, h_sampled)
#plt.show()
sns.kdeplot(np.array(h.ravel().cpu()), bw_adjust=0.1)
plt.show()


fig, ax = plt.subplots(8, 1, figsize=(8,32))
for i in range(N_H):
    sns.kdeplot(np.array(h[i, :].ravel().cpu()), bw_adjust=0.1, ax=ax[i])
plt.show()




