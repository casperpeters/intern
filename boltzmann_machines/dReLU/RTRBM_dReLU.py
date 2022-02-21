'scipt for the RTRBM with dReLU hidden unit potential'

import torch
from tqdm import tqdm


class RTRBM(object):

    def __init__(self, data, N_H=10, device=None):

        if device is None:
            self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        else:
            self.device = device

        self.dtype = torch.float
        self.V = data.float().to(self.device)
        self.dim = torch.tensor(self.V.shape).shape[0]

        if self.dim == 2:
            self.N_V, self.T = self.V.shape
        elif self.dim == 3:
            self.N_V, self.T, self.num_samples = self.V.shape
        else:
            raise ValueError("Data is not correctly defined: Use (N_V, T) or (N_V, T, num_samples) dimensions")

        self.N_H = N_H

        self.W = 0.01 * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.U = 0.01 * torch.randn(self.N_H, self.N_H, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(1, self.N_V, dtype=self.dtype, device=self.device)

        # Initial
        self.theta_p0 = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_m0 = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_p0 = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_m0 = torch.ones(self.N_H, dtype=self.dtype, device=self.device)

        # Normal
        self.theta_p = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_m = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_p = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_m = torch.ones(self.N_H, dtype=self.dtype, device=self.device)

        self.params = [self.W, self.U, self.b_V,
                       self.theta_p0, self.theta_m0, self.gamma_p0, self.gamma_m0,
                       self.theta_p, self.theta_m, self.gamma_p, self.gamma_m]

    def learn(self,
              n_epochs=1000,
              batchsize=128,
              CDk=10, PCD=False,
              lr=1e-3, lr_decay=None,
              sp=None, x=2,
              mom=0.9,
              wc=0.0002,
              AF=torch.sigmoid,
              disable_tqdm=False):

        global vt_k
        if self.dim == 2:
            num_batches = 1
            batchsize = 1
        elif self.dim == 3:
            num_batches = self.num_samples // batchsize

        Dparams = self.initialize_grad_updates()

        self.errors = torch.zeros(n_epochs, 1)
        self.disable = disable_tqdm
        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0

            for batch in range(0, num_batches):

                self.dparams = self.initialize_grad_updates()

                for i in range(0, batchsize):

                    if self.dim == 2:
                        v_data = self.V
                    elif self.dim == 3:
                        v_data = self.V[:, :, batch * batchsize + i]

                    # Forward
                    H_data = self.visible_to_expected_hidden(v_data)

                    # Perform contrastive divergence and compute model statistics
                    if PCD and epoch != 0:
                        # use last gibbs sample as input (Persistant Contrastive Divergence)
                        mean_H_cd, mean_v_cd, h_model, v_model = self.CD(v_model[:, :, -1],  CDk, AF=AF)
                    else:
                        # use data (normal Contrastive Divergence)
                        mean_H_cd, mean_v_cd, h_model, v_model = self.CD(v_data, H_data, CDk, AF=AF)

                    # Accumulate error
                    err += torch.sum((v_data - v_model[:, :, -1]) ** 2)

                    # Backpropagation, compute gradients and stores mean value
                    dparam = self.grad(v_data, H_data, h_model, v_model, mean_v_cd, mean_H_cd, CDk)
                    for i in range(len(dparam)): self.dparams[i] += dparam[i] / batchsize

                # Update gradients
                Dparams = self.update_grad(Dparams, lr=lr, mom=mom,wc=wc, sp=sp, x=x)

                # Apply constraints
                self.constraint()

            self.errors[epoch] = err / self.V.numel()

            if lr_decay is not None:
                if epoch % 10 == 0:
                    lr *= lr_decay

    def CD(self, v_data, H_data, CDk, AF=torch.sigmoid):

        h_model = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        H_model = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        v_model = torch.zeros(self.N_V, self.T, CDk, dtype=self.dtype, device=self.device)

        v_model[:, :, 0] = v_data.detach()
        H_model[:, :, 0] = self.visible_to_expected_hidden(v_model[:, :, 0])

        for kk in range(1, CDk):
            v_model[:, :, kk] = self.hidden_to_visible(h_model[:, :, kk], AF=AF)
            H_model[:, :, kk] = self.visible_to_expected_hidden(v_model[:, :, kk])
            h_model[:, :, kk] = self.visible_to_hidden(v_model[:, :, kk], H_data)

        mean_H_cd = torch.mean(H_model, 2)
        mean_v_cd = torch.mean(v_model, 2)

        return mean_H_cd, mean_v_cd, h_model, v_model

    def visible_to_expected_hidden(self, v):
        T = v.shape[1]
        H = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        for t in range(0, T):

            if t == 0:
                I = torch.matmul(v[:, t], self.W.T)
                tp = self.theta_p0
                tm = self.theta_m0
                gp = self.gamma_p0
                gm = self.gamma_m0
            elif t > 0:
                I = torch.matmul(v[:, t], self.W.T) #+ torch.matmul(self.U, H[:, t - 1])
                tp = self.theta_p
                tm = self.theta_m
                gp = self.gamma_p
                gm = self.gamma_m

            I_plus = (-I + tp) / torch.sqrt(gp)
            I_min = (I + tm) / torch.sqrt(gm)
            phi_plus = self.phi(I_plus)
            phi_min = self.phi(I_min)

            p_plus = torch.tensor(
                1 / (1 + (phi_min / torch.sqrt(gm)) / (phi_plus / torch.sqrt(gp))), \
                dtype=self.dtype, device=self.device)
            p_min = 1-p_plus

            H[:, t] = p_plus * ((I - tp) / gp + 1 / (torch.sqrt(gp) * phi_plus)) + \
                       p_min * ((I - tm) / gm - 1 / (torch.sqrt(gm) * phi_min))

            if torch.sum(torch.isnan(H[:, t])):
                a=1

        return H

    def visible_to_hidden(self, v, rt):
        """     Computes the hidden layer activations of the RNN.

            Parameters
            ----------
            vt : torch.Tensor
                The input data.

            Returns
            -------
            rt : torch.Tensor
                The hidden layer activations.

        """

        T = v.shape[1]
        h_sampled = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        for t in range(0, T):

            if t == 0:
                I = torch.matmul(v[:, t], self.W.T)
                tp = self.theta_p0
                tm = self.theta_m0
                gp = self.gamma_p0
                gm = self.gamma_m0
            elif t > 0:
                I = torch.matmul(v[:, t], self.W.T) + torch.matmul(self.U, rt[:, t - 1])
                tp = self.theta_p
                tm = self.theta_m
                gp = self.gamma_p
                gm = self.gamma_m

            I_plus = (-I + tp) / torch.sqrt(gp)
            I_min = (I + tm) / torch.sqrt(gm)
            phi_plus = self.phi(I_plus)
            phi_min = self.phi(I_min)

            p_plus = torch.tensor(1 / (1 + (phi_min / torch.sqrt(gm)) / (phi_plus / torch.sqrt(gp))), \
                                  dtype=self.dtype, device=self.device)

            sqrt2 = torch.tensor(1.41421356237, dtype=self.dtype, device=self.device)

            for i in range(self.N_H):  # for loop is obsolete
                if torch.isnan(p_plus[i]):
                    if torch.abs(I_plus[i]) > torch.abs(I_min[i]):
                        p_plus[i] = 1
                    else:
                        p_plus[i] = 0

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
                    h_sampled[i, t] = (out - I_plus[i]) / torch.sqrt(gp[i])
                else:
                    h_sampled[i, t] = (out + I_min[i]) / torch.sqrt(gm[i])

                if torch.isinf(out) | torch.isnan(out) | (rmax - rmin < 1e-14):
                    h_sampled[i, t] = 0

        if torch.sum(torch.isnan(h_sampled.ravel())):
            a = 1

        return h_sampled

    def hidden_to_visible(self, h, AF=torch.sigmoid):
        return torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

    def initialize_grad_updates(self):
        """
        Initializes a list of zero tensors with the same shape as the model parameters.

        Parameters:
            self (torch.nn.Module): The model.

        Returns:
            list: A list of zero tensors with the same shape as the model parameters. """

        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad_dReLU(self, I_data, I_model, t=None):

        if t == 0:
            tp = self.theta_p0
            tm = self.theta_m0
            gp = self.gamma_p0
            gm = self.gamma_m0
            
        elif t == None and t != 0:
            tp = self.theta_p
            tm = self.theta_m
            gp = self.gamma_p
            gm = self.gamma_m

        # Data
        I_d = I_data
        I_plus = (-I_d + tp) / torch.sqrt(gp)
        I_min = (I_d + tm) / torch.sqrt(gm)
        phi_plus_d = self.phi(I_plus) / torch.sqrt(gp)
        phi_min_d = self.phi(I_min) / torch.sqrt(gm)
        p_plus_d = 1 / (1 + (phi_min_d / phi_plus_d))
        p_min_d = 1 - p_plus_d

        # Model
        I_m = I_model
        I_plus = (-I_m + tp) / torch.sqrt(gp)
        I_min = (I_m + tm) / torch.sqrt(gm)
        phi_plus_m = self.phi(I_plus) / torch.sqrt(gp)
        phi_min_m = self.phi(I_min) / torch.sqrt(gm)
        p_plus_m = 1 / (1 + (phi_min_m / phi_plus_m))
        p_min_m = 1 - p_plus_m

        # positive - negative gradient
        dtheta_p = - ((p_plus_d * ((I_d - tp) / gp + 1 / (torch.sqrt(gp) * phi_plus_d))) - \
                      (p_plus_m * ((I_m - tp) / gp + 1 / (torch.sqrt(gp) * phi_plus_m))))

        dtheta_m = - ((p_min_d * ((I_d - tm) / gm - 1 / (torch.sqrt(gm) * phi_min_d))) - \
                      (p_min_m * ((I_m - tm) / gm - 1 / (torch.sqrt(gm) * phi_min_m))))

        dgamma_p = -1 / 2 * ((p_plus_d * (1 / gp + ((I_d - tp)
                                                              / gp) ** 2 + (I_d - tp) / (
                                                  gp * phi_plus_d))) -

                             (p_plus_m * (1 / gp + ((I_m - tp)
                                                              / gp) ** 2 + (I_m - tp) / (
                                                  gp * phi_plus_m))))

        dgamma_m = -1 / 2 * ((p_min_d * (1 / gm + ((I_d - tm)
                                                             / gm) ** 2 - (I_d - tm) / (
                                                 gm * phi_min_d))) -

                             (p_min_m * (1 / gm + ((I_m - tm)
                                                             / gm) ** 2 - (I_m - tm) / (
                                                 gm * phi_min_m))))

        if torch.sum(torch.isnan(dtheta_p)):
            a=1
        if torch.sum(torch.isnan(dtheta_m)):
            a=1
        if torch.sum(torch.isnan(dgamma_p)):
            a=1
        if torch.sum(torch.isnan(dgamma_m)):
            a=1

        return dtheta_p, dtheta_m, dgamma_p, dgamma_m

    def drt(self, I,  derivative='rt'):
        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I + self.theta_m) / torch.sqrt(self.gamma_m)
        phi_p = self.phi(I_plus)
        phi_m = self.phi(I_min)
        Zp = phi_p / torch.sqrt(self.gamma_p)
        Zm = phi_m / torch.sqrt(self.gamma_m)
        p_plus = 1 / (1 + (Zp / Zm))
        p_min = 1 - p_plus

        dphi_p = self.dphi(I_plus)
        dphi_m = self.dphi(I_min)

        gp0 = self.gamma_p0
        gm0 = self.gamma_m0
        tp0 = self.theta_p0
        tm0 = self.theta_m0

        gp = self.gamma_p
        gm = self.gamma_m
        tp = self.theta_p
        tm = self.theta_m


        if derivative =='rt':
            drt = 1 / (gp * gm * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  (gm ** 2 * phi_p ** 2 + gp ** 2 * phi_m ** 2 + \
                   torch.sqrt(gm) * phi_m * dphi_p * (-gp * tm + gm * tp + I * (-gm + gp)) + \
                   (gm + gp) * (gm * dphi_p - gp * dphi_m) + \
                   torch.sqrt(gp) * phi_p * (torch.sqrt(gm) * (gm + gp) * phi_m + \
                                             (-gp * tm + gm * tp + I * (-gm + gp)) * dphi_m)
                   ) # U is later added line 515
            if torch.sum(torch.isnan(drt)):
                a=1
                
        if derivative == 'U':
            drt = 1 / (gp * gm * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  (gm ** 2 * phi_p ** 2 + gp ** 2 * phi_m ** 2 + \
                   torch.sqrt(gm) * phi_m * dphi_p * (-gp * tm + gm * tp + I * (-gm + gp)) + \
                   (gm + gp) * (gm * dphi_p - gp * dphi_m) + \
                   torch.sqrt(gp) * phi_p * (torch.sqrt(gm) * (gm + gp) * phi_m + \
                                             (-gp * tm + gm * tp + I * (-gm + gp)) * dphi_m)
                   ) # rt is later added line 536
            if torch.sum(torch.isnan(drt)):
                a = 1
                
        if derivative == 'W':
            drt = 1 / (gp * gm * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  (gm ** 2 * phi_p ** 2 + gp ** 2 * phi_m ** 2 + \
                   torch.sqrt(gm) * phi_m * dphi_p * (-gp * tm + gm * tp + I * (-gm + gp)) + \
                   (gm + gp) * (gm * dphi_p - gp * dphi_m) + \
                   torch.sqrt(gp) * phi_p * (torch.sqrt(gm) * (gm + gp) * phi_m + \
                                             (-gp * tm + gm * tp + I * (-gm + gp)) * dphi_m)
                   ) # vt is later added line 545            
            if torch.sum(torch.isnan(drt)):
                a=1
                
        if derivative == 'theta_p0':

            drt = 1 / (gp0 * torch.sqrt(gm0) * (torch.sqrt(gm0) * phi_p + torch.sqrt(gp0) * phi_m) ** 2) * \
                  -(gm0 ** (3 / 2) * phi_p ** 2 + gm0 * torch.sqrt(gp0) * phi_p * phi_m + \
                   (torch.sqrt(gm0) * (gm0 + gp0) + (-gp0 * tm0 + gm0 * tp0 + I*(-gm0 + gp0)) * phi_m) * dphi_p
                   )
            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative == 'theta_p':
            drt = 1 / (gp * torch.sqrt(gm) * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  -(gm ** (3 / 2) * phi_p ** 2 + gm * torch.sqrt(gp) * phi_p * phi_m + \
                   (torch.sqrt(gm) * (gm + gp) + (-gp * tm + gm * tp + I*(-gm + gp)) * phi_m) * dphi_p
                   )
            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative == 'theta_m0':
            drt = 1 / (torch.sqrt(gp0) * gm0 * (torch.sqrt(gm0) * phi_p + torch.sqrt(gp0) * phi_m) ** 2) * \
                  (-gp0 * phi_m * (torch.sqrt(gm0) * phi_p + torch.sqrt(gp0) * phi_m) + \
                   (torch.sqrt(gp0) * (gm0 + gp0) + (gp0 * tm0 - gm0 * tp0 + I*(gm0 - gp0)) * phi_p) * dphi_m
                   )
            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative == 'theta_m':
            drt = 1 / (torch.sqrt(gp) * gm * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  (-gp * phi_m * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) + \
                   (torch.sqrt(gp) * (gm + gp) + (gp * tm - gm * tp + I*(gm - gp)) * phi_p) * dphi_m
                   )
            if torch.sum(torch.isnan(drt)):
                a=1
                
        if derivative == 'gamma_p0':
            drt = 1 / (2 * torch.sqrt(gm0) * (torch.sqrt(gm0) * gp0 * phi_p + gp0 ** (3/2) * phi_m) ** 2) * \
                  (2 * gm0 ** (3/2) * (tp0 - I) * phi_p ** 2 - 2 * gm0 * gp0 * phi_m + \
                   torch.sqrt(gp0) * phi_p * (
                            torch.sqrt(gm0) * (-gm0 + gp0) + (-gp0 * tm0 + 3 * gm0 * tp0 + I*(-3 * gm0 + gp0)) * phi_m) + \
                   (tp0 - I) * (torch.sqrt(gm0) * (gm0 + gp0) + (-gp0 * tm0 + gm0 * tp0 + I*(-gm0 + gp0)) * phi_m) * dphi_p
                   )
            if torch.sum(torch.isnan(drt)):
                a=1
                
        if derivative == 'gamma_p':
            drt = 1 / (2 * torch.sqrt(gm) * (torch.sqrt(gm) * gp * phi_p + gp ** (3 / 2) * phi_m) ** 2) * \
                  (2 * gm ** (3 / 2) * (tp - I) * phi_p ** 2 - 2 * gm * gp * phi_m + \
                   torch.sqrt(gp) * phi_p * (
                               torch.sqrt(gm) * (-gm + gp) + (-gp * tm + 3 * gm * tp +I*(-3 * gm + gp)) * phi_m) + \
                   (tp - I) * (torch.sqrt(gm) * (gm + gp) + (-gp * tm + gm * tp + I*(-gm + gp)) * phi_m) * dphi_p
                   )
            if torch.sum(torch.isnan(drt)):
                a=1
                
        if derivative == 'gamma_m0':
            drt = - 1 / (2 * torch.sqrt(gp0) * (torch.sqrt(gp0) * gm0 * phi_m + gm0 ** (3 / 2) * phi_p) ** 2) * \
                  (torch.sqrt(gp0) * (torch.sqrt(gm0) * (-gm0 + gp0) * phi_m + 2 * gp0 * (-tm0 + I) * phi_m ** 2 + \
                                     (gm0 + gp0) * (tm0 - I) * dphi_m) + \
                   phi_p * (2 * gm0 * gp0 + torch.sqrt(gm0) * (-3 * gp0 * tm0 + gm0 * tp0 - gm0 * I + 3 * gp0 * I) * phi_m + \
                            (tm0 - I) * (gp0 * tm0 - gm0 * tp0 + I * (gm0 - gp0)) * dphi_m)
                   )
            if torch.sum(torch.isnan(drt)):
                a=1
                
        if derivative == 'gamma_m':
            drt = - 1 / (2 * torch.sqrt(gp) * (torch.sqrt(gp) * gm * phi_m + gm ** (3 / 2) * phi_p) ** 2) * \
                  (torch.sqrt(gp) * (torch.sqrt(gm) * (-gm + gp) * phi_m + 2 * gp * (-tm + I) * phi_m ** 2 + \
                                     (gm + gp) * (tm - I) * dphi_m) + \
                   phi_p * (2 * gm * gp + torch.sqrt(gm) * (-3 * gp * tm + gm * tp - gm * I + 3 * gp * I) * phi_m + \
                            (tm - I) * (gp * tm - gm * tp + I*(gm - gp)) * dphi_m)
                  )

            if torch.sum(torch.isnan(drt)):
                a=1

        return drt

    def grad(self, vt, rt, ht_k, vt_k, barvt, barht, CDk):
        """
            Computes the gradient of the parameters of the RTRBM.

            :param vt: The visible layer activations of the RBM (i.e. the input)
            :param rt: The hidden layer activations of the RBM (i.e. the output)
            :param ht_k: The hidden layer activations of the top RBM
            :param vt_k: The visible layer activations of the top RBM
            :param barht: The mean activations of the hidden layer
            :param barvt: The mean activations of the visible layer
            :param CDk: The number of Gibbs sampling steps used to compute the negative phase in CD-k
            :return: A list containing the gradient of the parameters of the RBM """


        # Gradients for HU potential
        param = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        param0 = [param, param, param, param]# dtheta_p0, dtheta_m0, dgamma_p0, dgamma_m0
        param = [param, param, param, param]# dtheta_p, dtheta_m, dgamma_p, dgamma_m

        drt_drt_min_1 = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        dr_dU = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        dr_dW = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        dr_dtheta_p = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        dr_dtheta_m = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        dr_dgamma_p = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        dr_dgamma_m = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)

        for t in range(self.T):
            if t==0:
                I_data = torch.matmul(vt[:, t], self.W.T)
                I_model = torch.matmul(vt_k[:, t, -1], self.W.T)
                param_t = self.grad_dReLU(I_data, I_model, t=0)
                for i in range(len(param_t)): param0[i] = param_t[i]

            elif t>0:
                I_data = torch.matmul(vt[:, t], self.W.T) + torch.matmul(self.U, rt[:, t - 1])
                I_model = torch.matmul(vt_k[:, t, -1], self.W.T) + torch.matmul(self.U, rt[:, t - 1])
                param_t = self.grad_dReLU(I_data, I_model)
                for i in range(len(param_t)): param[i] += param_t[i]

            dr_dtheta_p[:, t] = self.drt(I_data, derivative='theta_p')
            dr_dtheta_m[:, t] = self.drt(I_data, derivative='theta_m')
            dr_dgamma_p[:, t] = self.drt(I_data, derivative='gamma_p')
            dr_dgamma_m[:, t] = self.drt(I_data, derivative='gamma_m')

            drt_drt_min_1[:, t] = self.drt(I_data, derivative='rt')
            dr_dU[:, t] = self.drt(I_data, derivative='U')
            dr_dW[:, t] = self.drt(I_data, derivative='W')

        dtheta_p0, dtheta_m0, dgamma_p0, dgamma_m0 = param0
        dtheta_p, dtheta_m, dgamma_p, dgamma_m = param

        # Backpropagation
        Dt = torch.zeros(self.N_H, self.T + 1, dtype=self.dtype, device=self.device)

        for t in range(self.T - 1, -1, -1): # begin, stop, step
            Dt[:, t] = torch.matmul(self.U.T, (Dt[:, t + 1] * drt_drt_min_1[:, t] + (rt[:, t] - barht[:, t])))

        # Gradients for HU potential
        dtheta_p += torch.sum(Dt[:, 2:self.T] * dr_dtheta_p[:, 1:self.T - 1])
        dtheta_m += torch.sum(Dt[:, 2:self.T] * dr_dtheta_m[:, 1:self.T - 1])
        dgamma_p += torch.sum(Dt[:, 2:self.T] * dr_dgamma_p[:, 1:self.T - 1])
        dgamma_m += torch.sum(Dt[:, 2:self.T] * dr_dgamma_m[:, 1:self.T - 1])

        # Gradient for HU potential initial value
        dtheta_p0 += Dt[:, 1] * dr_dtheta_p[:, 0]
        dtheta_m0 += Dt[:, 1] * dr_dtheta_m[:, 0]
        dgamma_p0 += Dt[:, 1] * dr_dgamma_p[:, 0]
        dgamma_m0 += Dt[:, 1] * dr_dgamma_m[:, 0]

        # Gradient for visible field
        db_V = torch.sum(vt - barvt, 1)

        # Gradient for W

        dW_1 = torch.sum(
            (Dt[:, 1:self.T] * dr_dW[:, 0:self.T-1]).unsqueeze(1).repeat(1, self.N_V, 1) *
            vt[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)
        dW_2 = torch.sum(rt.unsqueeze(1).repeat(1, self.N_V, 1) * vt.unsqueeze(0).repeat(self.N_H, 1, 1), 2) - \
               torch.sum(
                   torch.sum(ht_k.unsqueeze(1).repeat(1, self.N_V, 1, 1) * vt_k.unsqueeze(0).repeat(self.N_H, 1, 1, 1),
                             3), 2) / CDk
        dW = dW_1 + dW_2

        # Gradient for U
        dU = torch.sum((Dt[:, 2:self.T + 1] * (dr_dU[:, 1:self.T]) + rt[:, 1:self.T] - barht[:, 1:self.T]).unsqueeze(
            1).repeat(1, self.N_H, 1) * rt[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)

        if torch.sum(torch.isnan(dtheta_p)):
            a = 1
        if torch.sum(torch.isnan(dtheta_m)):
            a = 1
        if torch.sum(torch.isnan(dgamma_p)):
            a = 1
        if torch.sum(torch.isnan(dgamma_m)):
            a = 1

        if torch.sum(torch.isnan(dtheta_p0)):
            a=1
        if torch.sum(torch.isnan(dtheta_m0)):
            a=1
        if torch.sum(torch.isnan(dgamma_p0)):
            a=1
        if torch.sum(torch.isnan(dgamma_m0)):
            a=1

        if torch.sum(torch.isnan(Dt)):
            a=1
        if torch.sum(torch.isnan(db_V)):
            a=1
        if torch.sum(torch.isnan(dW)):
            a=1
        if torch.sum(torch.isnan(dU)):
            a=1

        del Dt

        return [dW, dU, db_V, dtheta_p0, dtheta_m0, dgamma_p0, dgamma_m0, dtheta_p, dtheta_m, dgamma_p, dgamma_m]

    def update_grad(self, Dparams, lr=1e-3, mom=0,wc=0, x=2, sp=None):

        dW, dU, db_V, dtheta_p0, dtheta_m0, dgamma_p0, dgamma_m0, dtheta_p, dtheta_m, dgamma_p, dgamma_m = self.dparams
        DW, DU, Db_V, Dtheta_p0, Dtheta_m0, Dgamma_p0, Dgamma_m0, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Db_V = mom * Db_V + lr * db_V
        Dtheta_p0 = mom * Dtheta_p0 + lr * dtheta_p0
        Dtheta_m0 = mom * Dtheta_m0 + lr * dtheta_m0
        Dgamma_p0 = mom * Dgamma_p0 + lr * dgamma_p0
        Dgamma_m0 = mom * Dgamma_m0 + lr * dgamma_m0

        Dtheta_p = mom * Dtheta_p + lr * dtheta_p
        Dtheta_m = mom * Dtheta_m + lr * dtheta_m
        Dgamma_p = mom * Dgamma_p + lr * dgamma_p
        Dgamma_m = mom * Dgamma_m + lr * dgamma_m

        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V), \
                                     [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, DU, Db_V, Dtheta_p0, Dtheta_m0, Dgamma_p0, Dgamma_m0, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m]

        for i in range(len(self.params)): self.params[i] += Dparams[i]

        return [DW, DU, Db_V, Dtheta_p0, Dtheta_m0, Dgamma_p0, Dgamma_m0, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m]

    def constraint(self, threshold = 1e-2):
        for p in [self.gamma_p, self.gamma_m, self.gamma_p0, self.gamma_m0]:
            idx = p < threshold
            p[idx] = threshold

    def phi(self, x):

        sqrtpi = torch.tensor(1.77245385091, dtype=self.dtype,
                                   device=self.device)
        sqrt2 = torch.tensor(1.41421356237, dtype=self.dtype,
                                   device=self.device)
        out = torch.zeros(torch.numel(x), dtype=self.dtype, device=self.device)
        size = torch.numel(x)
        if size > 1:
            for i in range(torch.numel(x)):
                if x[i] < -5.85:
                    out[i] = 2 * torch.exp(x[i] ** 2 / 2)
                elif x[i] > 5.85:
                    out[i] = 1/x[i] - 1/x[i] ** 3 + 3 / x[i] ** 5
                else:
                    out[i] = torch.exp(x[i] ** 2 / 2) * (1 - torch.erf(x[i]/sqrt2)) * sqrtpi / sqrt2
        elif size == 1:
            if x < -5.85:
                out = sqrt2 * sqrtpi * torch.exp(x ** 2 / 2)
            elif x > 5.85:
                out = 1/x - 1/x ** 3 + 3 / x ** 5
            else:
                out = torch.exp(x ** 2 / 2) * (1 - torch.erf(x/sqrt2)) * sqrtpi / sqrt2
        if torch.sum(torch.isnan(out)):
            a=1
        return out

    def dphi(self, x):
        out = x * self.phi(x) - 1
        if torch.sum(torch.isnan(out)):
            a=1
        return out

    def visible_to_hidden_infer(self, v, ht_min_one=None):
        """     Computes the hidden layer activations of the RNN.

            Parameters
            ----------
            vt : torch.Tensor
                The input data.

            Returns
            -------
            rt : torch.Tensor
                The hidden layer activations.

        """
        h_sampled = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)

        if ht_min_one is None:
            I = torch.matmul(v, self.W.T) + self.b_init[0]
        elif ht_min_one is not None:
            I = torch.matmul(v, self.W.T) + torch.matmul(self.U, ht_min_one)

        I_plus = (-I + self.theta_p) / torch.sqrt(self.gamma_p)
        I_min = (I + self.theta_m) / torch.sqrt(self.gamma_m)
        phi_plus = self.phi(I_plus)
        phi_min = self.phi(I_min)

        p_plus = torch.tensor(
            1 / (1 + (phi_min / torch.sqrt(self.gamma_m)) / (phi_plus / torch.sqrt(self.gamma_p))), \
            dtype=self.dtype, device=self.device)

        sqrt2 = torch.tensor(1.41421356237, dtype=self.dtype, device=self.device)

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
                h_sampled[i] = (out - I_plus[i]) / torch.sqrt(self.gamma_p[i])
            else:
                h_sampled[i] = (out + I_min[i]) / torch.sqrt(self.gamma_m[i])

            if torch.isinf(out) | torch.isnan(out) | (rmax - rmin < 1e-14):
                h_sampled[i] = 0
        return h_sampled

    def sample(self,
               v_start,
               AF=torch.sigmoid,
               chain=50,
               pre_gibbs_k=100,
               gibbs_k=20,
               mode=1,
               disable_tqdm=False):

        v_sampled = torch.zeros(self.N_V, chain, dtype=self.dtype, device=self.device)
        h_sampled = torch.zeros(self.N_H, chain, dtype=self.dtype, device=self.device)

        v_sampled[:, 0] = v_start.detach().clone().to(self.device)
        h_sampled[:, 0] = self.visible_to_hidden_infer(v_sampled[:, 0])

        for t in tqdm(range(1, chain), disable=disable_tqdm):
            v = v_sampled[:, t - 1]

            # it is important to keep the burn-in inside the chain loop, because we now have time-dependency
            for kk in range(pre_gibbs_k):
                h = self.visible_to_hidden_infer(v, ht_min_one=h_sampled[:, t - 1])
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V))[0]

            vt_k = torch.zeros(self.N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = self.visible_to_hidden_infer(v, ht_min_one=h_sampled[:, t - 1])
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V))[0]
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                v_sampled[:, t] = vt_k[:, -1]
            if mode == 2:
                v_sampled[:, t] = torch.mean(vt_k, 1)

            h_sampled[:, t] = self.visible_to_hidden_infer(v, ht_min_one=h_sampled[:, t - 1])

        return v_sampled, h_sampled


# This is an example run:
#"""

import sys
sys.path.append(r'D:\RU\OneDrive\Intern\rtrbm_master')
from utils.plots import *
from data.mock_data import *
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

N_V, N_H, T = 16, 8, 32
data = create_BB(N_V=N_V, T=T, n_samples=8, width_vec=[5], velocity_vec=[2], boundary=False)

rtrbm = RTRBM(data, N_H, device="cpu")
rtrbm.learn(batchsize=2, n_epochs=100, lr=0.01, lr_decay=0.9)

vt_infer, rt_infer = rtrbm.sample(v_start=data[:, 4, 0])

sns.heatmap(vt_infer.cpu())
plt.show()

plt.plot(rtrbm.errors.cpu())
plt.show()


#"""

