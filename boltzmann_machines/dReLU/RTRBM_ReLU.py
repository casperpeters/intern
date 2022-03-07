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

        self.W = 0.01/self.N_V * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.U_p = 0.01/self.N_H * torch.randn(self.N_H, self.N_H, dtype=self.dtype, device=self.device)
        self.U_m = 0.01/self.N_H * torch.randn(self.N_H, self.N_H, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(1, self.N_V, dtype=self.dtype, device=self.device)

        # Initial
        self.b_init = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)

        # Normal
        self.theta_p = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.theta_m = torch.zeros(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_p = torch.ones(self.N_H, dtype=self.dtype, device=self.device)
        self.gamma_m = torch.ones(self.N_H, dtype=self.dtype, device=self.device)

        self.params = [self.W, self.U_p, self.U_m, self.b_V,
                       self.b_init,
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
                    r_p_data, r_m_data = self.visible_to_expected_hidden(v_data)

                    # use data (normal Contrastive Divergence)
                    mean_rt_p_model, mean_rt_m_model, mean_vt_model, h_model, v_model = self.CD(v_data, r_p_data, r_m_data, CDk, AF=AF)

                    # Accumulate error
                    err += torch.sum((v_data - v_model[:, :, -1]) ** 2)

                    # Backpropagation, compute gradients and stores mean value
                    dparam = self.grad(v_data, r_p_data, r_m_data, h_model, v_model, mean_vt_model, mean_rt_p_model, mean_rt_m_model, CDk)
                    for i in range(len(dparam)): self.dparams[i] += dparam[i] / batchsize

                # Update gradients
                Dparams = self.update_grad(Dparams, lr=lr, mom=mom,wc=wc, sp=sp, x=x)

                # Apply constraints
                self.constraint()

            self.errors[epoch] = err / self.V.numel()

            if lr_decay is not None:
                if epoch % 10 == 0:
                    lr *= lr_decay

    def CD(self, v_data, r_p_data, r_m_data, CDk, AF=torch.sigmoid):

        h_model = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        rt_p_model = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        rt_m_model = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        v_model = torch.zeros(self.N_V, self.T, CDk, dtype=self.dtype, device=self.device)

        v_model[:, :, 0] = v_data.detach()
        rt_p_model[:, :, 0], rt_m_model[:, :, 0] = self.visible_to_expected_hidden(v_model[:, :, 0])
        h_model[:, :, 0] = self.visible_to_hidden(v_model[:, :, 0], r_p_data, r_m_data)

        for kk in range(1, CDk):
            v_model[:, :, kk] = self.hidden_to_visible(h_model[:, :, kk], AF=AF)
            rt_p_model[:, :, kk], rt_m_model[:, :, kk] = self.visible_to_expected_hidden(v_model[:, :, kk])
            h_model[:, :, kk] = self.visible_to_hidden(v_model[:, :, kk], r_p_data, r_m_data)

        mean_r_p_model = torch.mean(rt_p_model, 2)
        mean_r_m_model = torch.mean(rt_m_model, 2)
        mean_vtm = torch.mean(v_model, 2)

        return mean_r_p_model, mean_r_m_model, mean_vtm, h_model, v_model

    def visible_to_expected_hidden(self, v):
        T = v.shape[1]
        rt_p = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        rt_m = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)

        tp = self.theta_p.detach().clone()
        tm = self.theta_m.detach().clone()
        gp = self.gamma_p.detach().clone()
        gm = self.gamma_m.detach().clone()

        for t in range(0, T):

            I = torch.matmul(v[:, t], self.W.T)
            if t == 0:
                IU_p = self.b_init_p[0].detach().clone()
                IU_m = self.b_init_m[0].detach().clone()
            elif t > 0:
                IU_p = torch.matmul(self.U_p, rt_p[:, t - 1])
                IU_m = torch.matmul(self.U_m, rt_m[:, t - 1])

            I_plus = (-I + tp + IU_p) / torch.sqrt(gp)
            I_min = (I + tm + IU_m) / torch.sqrt(gm)

            phi_plus = self.phi(I_plus)
            phi_min = self.phi(I_min)

            p_plus = torch.tensor(
                1 / (1 + (phi_min / torch.sqrt(gm)) / (phi_plus / torch.sqrt(gp))), \
                dtype=self.dtype, device=self.device)

            for i in range(self.N_H):
                if torch.isnan(p_plus[i]):
                    if torch.abs(I_plus[i]) > torch.abs(I_min[i]):
                        p_plus[i] = 1
                    else:
                        p_plus[i] = 0
            p_plus[p_plus > 1] = 1
            p_plus[p_plus < 0] = 0

            p_min = 1 - p_plus
            rt_p[:, t] = p_plus * ((I - tp - IU_p) / gp + 1 / (torch.sqrt(gp) * phi_plus))
            rt_m[:, t] = p_min * ((I - tm - IU_m) / gm - 1 / (torch.sqrt(gm) * phi_min))

            if torch.sum(torch.isnan(rt_p[:, t])):
                a=1
            if torch.sum(torch.isnan(rt_m[:, t])):
                a=1

        return rt_p, rt_m

    def visible_to_hidden(self, v, rt_p, rt_m):
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
        tp = self.theta_p.detach().clone()
        tm = self.theta_m.detach().clone()
        gp = self.gamma_p.detach().clone()
        gm = self.gamma_m.detach().clone()

        for t in range(0, T):

            I = torch.matmul(v[:, t], self.W.T)
            if t == 0:
                IU_p = self.b_init_p[0].detach().clone()
                IU_m = self.b_init_m[0].detach().clone()
            elif t > 0:
                IU_p = torch.matmul(self.U, rt_p[:, t - 1])
                IU_m = torch.matmul(self.U, rt_m[:, t - 1])

            I_plus = (-I + tp + IU_p) / torch.sqrt(gp)
            I_min = (I + tm + IU_m) / torch.sqrt(gm)

            phi_plus = self.phi(I_plus)
            phi_min = self.phi(I_min)

            p_plus = torch.tensor(1 / (1 + (phi_min / torch.sqrt(gm)) / (phi_plus / torch.sqrt(gp))), \
                                  dtype=self.dtype, device=self.device)

            sqrt2 = torch.tensor(1.41421356237, dtype=self.dtype, device=self.device)

            for i in range(self.N_H):
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

    def grad_dReLU(self, I_data, I_model, rt_p_data=None, rt_m_data=None, rt_p_model=None, rt_m_model=None): # take for rt=rt[:, t-1]


        tp = self.theta_p.detach().clone()
        tm = self.theta_m.detach().clone()
        gp = self.gamma_p.detach().clone()
        gm = self.gamma_m.detach().clone()

        if rt_p_data == None or rt_m_data == None or rt_p_model == None or rt_m_model == None:
            IU_p_data = self.b_init_p[0].detach().clone()
            IU_m_data = self.b_init_m[0].detach().clone()
            IU_p_model = self.b_init_p[0].detach().clone()
            IU_m_model = self.b_init_m[0].detach().clone()

        if rt_p_data != None or rt_m_data != None or rt_p_model != None or rt_m_model != None:
            IU_p_data = torch.matmul(self.U_p, rt_p_data) # needs to be rt[:, t-1]
            IU_m_data = torch.matmul(self.U_m, rt_m_data)

            IU_p_model = torch.matmul(self.U_p, rt_p_model)  # needs to be rt[:, t-1]
            IU_m_model = torch.matmul(self.U_m, rt_m_model)


        # Data
        I_d = I_data
        I_plus = (-I_d + tp + IU_p_data) / torch.sqrt(gp)
        I_min = (I_d + tm + IU_m_data) / torch.sqrt(gm)
        phi_plus_d = self.phi(I_plus) / torch.sqrt(gp)
        phi_min_d = self.phi(I_min) / torch.sqrt(gm)
        p_plus_d = 1 / (1 + (phi_min_d / phi_plus_d))
        p_min_d = 1 - p_plus_d

        # Model
        I_m = I_model
        I_plus = (-I_m + tp + IU_p_model) / torch.sqrt(gp)
        I_min = (I_m + tm + IU_m_model) / torch.sqrt(gm)
        phi_plus_m = self.phi(I_plus) / torch.sqrt(gp)
        phi_min_m = self.phi(I_min) / torch.sqrt(gm)
        p_plus_m = 1 / (1 + (phi_min_m / phi_plus_m))
        p_min_m = 1 - p_plus_m

        # positive - negative gradient
        dtheta_p = - ((p_plus_d * ((I_d - tp - IU_p_data) / gp + 1 / (torch.sqrt(gp) * phi_plus_d))) - \
                      (p_plus_m * ((I_m - tp - IU_p_model) / gp + 1 / (torch.sqrt(gp) * phi_plus_m))))

        dtheta_m = - ((p_min_d * ((I_d - tm - IU_m_data) / gm - 1 / (torch.sqrt(gm) * phi_min_d))) - \
                      (p_min_m * ((I_m - tm - IU_m_model) / gm - 1 / (torch.sqrt(gm) * phi_min_m))))

        dgamma_p = -1 / 2 * ((p_plus_d * (1 / gp + ((I_d - tp - IU_p_data)
                                                              / gp) ** 2 + (I_d - tp - IU_p_data) / (
                                                  gp * phi_plus_d))) -

                             (p_plus_m * (1 / gp + ((I_m - tp - IU_p_model)
                                                              / gp) ** 2 + (I_m - tp - IU_p_model) / (
                                                  gp * phi_plus_m))))

        dgamma_m = -1 / 2 * ((p_min_d * (1 / gm + ((I_d - tm - IU_m_data)
                                                             / gm) ** 2 - (I_d - tm - IU_m_data) / (
                                                 gm * phi_min_d))) -

                             (p_min_m * (1 / gm + ((I_m - tm - IU_m_model)
                                                             / gm) ** 2 - (I_m - tm - IU_m_model) / (
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

    def drt(self, I, rt_p=None, rt_m=None, derivative='rt'): # take for rt=rt[:, t-1]

        gp = self.gamma_p.detach().clone()
        gm = self.gamma_m.detach().clone()
        tp = self.theta_p.detach().clone()
        tm = self.theta_m.detach().clone()

        if rt_p == None or rt_m == 0:
            IU_p = self.b_init_p[0].detach().clone()
            IU_m = self.b_init_m[0].detach().clone()
        elif rt_p != None or rt_m != None:
            IU_p = torch.matmul(self.U_p, rt_p) # needs to be rt[:, t-1]
            IU_m = torch.matmul(self.U_m, rt_m) # needs to be rt[:, t-1]

        I_plus = (-I + tp + IU_p) / torch.sqrt(gp)
        I_min = (I + tm + IU_m) / torch.sqrt(gm)
        phi_p = self.phi(I_plus)
        phi_m = self.phi(I_min)
        Zp = phi_p / torch.sqrt(gp)
        Zm = phi_m / torch.sqrt(gm)
        p_plus = 1 / (1 + (Zp / Zm))
        p_min = 1 - p_plus

        dphi_p = self.dphi(I_plus)
        dphi_m = self.dphi(I_min)



        if derivative =='rt_p':
            drt = (-1 - dphi_p / phi_p**2) / gp # dU_p chain rule is later added
            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative =='rt_m':
            drt = (1 - dphi_m/phi_m**2) / gm # dU_m chain rule is later added
            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative =='U_p':
            drt = (-1 - dphi_p / phi_p ** 2) / gp  # drt_p chain rule is later added
            if torch.sum(torch.isnan(drt)):
                a = 1

        if derivative =='U_m':
            drt = (-1 - dphi_m / phi_m ** 2) / gm  # drt_m chain rule is later added
            if torch.sum(torch.isnan(drt)):
                a = 1

        if derivative =='b_init_p':
            drt = (-1 - dphi_p / phi_p ** 2) / gp
            if torch.sum(torch.isnan(drt)):
                a = 1

        if derivative =='b_init_m':
            drt = (-1 - dphi_m / phi_m ** 2) / gm
            if torch.sum(torch.isnan(drt)):
                a = 1

        if derivative == 'W':
            drt =   # vt is later added line 515
            if torch.sum(torch.isnan(drt)):
                a = 1

        if derivative == 'theta_p':
            drt = - 1 / (torch.sqrt(gm) * gp * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  (gm ** (3/2) * phi_p ** 2 + gm * torch.sqrt(gp) * phi_p * phi_m +\
                   dphi_p * (torch.sqrt(gm) * (gm + gp) + (gm * tp - gp * tm + (gm - gp) * (IU - I)) * phi_m)
                  )

            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative == 'theta_m':
            drt = 1 / (torch.sqrt(gp) * gm * (torch.sqrt(gm) * phi_p + torch.sqrt(gp) * phi_m) ** 2) * \
                  torch.sqrt(gp) * ((-gp * phi_m ** 2 + (gm + gp) * dphi_m) -\
                  phi_p * (torch.sqrt(gm) * gp * phi_m +\
                        (gm * tp - gp * tm + (gm - gp) * (IU - I)) * dphi_m)
                  )
            if torch.sum(torch.isnan(drt)):
                a=1

        if derivative == 'gamma_p':
            drt = 1 / (2 * torch.sqrt(gm) * (gp ** (3/2) * phi_m + torch.sqrt(gm) * gp * phi_p) ** 2) * (
                2 * gm ** (3/2) * (tp + IU - I) * phi_p ** 2 - 2 * gm * gp * phi_m + \
                torch.sqrt(gp) * phi_p * (torch.sqrt(gm) * (gp - gm) + phi_m * \
                                          (-gp * tm + 3 * gm * gp + (3 * gm - gp) * (IU -I))) + \
                (tp + IU - I) * dphi_p * (torch.sqrt(gm) * (gm + gp) + phi_m * \
                                           (-gp * tm + gm * gp + (gm - gp) * (IU - I)))
                )
            if torch.sum(torch.isnan(drt)):
                a = 1

        if derivative == 'gamma_m':
            drt = 1 / (2 * torch.sqrt(gp) * (gm ** (3 / 2) * phi_p + torch.sqrt(gp) * gm * phi_m) ** 2) * (
                 torch.sqrt(gp) * (torch.sqrt(gm) * (gm - gp) * phi_m + 2 * gp * (tm + IU - I) * phi_m ** 2 - \
                                   (gm + gp) * (tm + IU - I) * dphi_m) + \
                phi_p * (-2 * gm * gp + torch.sqrt(gm) * (3 * gp * tm - gm * tp - (gm - 3 * gp) * (IU - I)) * phi_m - \
                         (gp * tm - gm * tp - (gm - gp) * (IU - I)) * (tm + IU - I) * dphi_m)
                )
            if torch.sum(torch.isnan(drt)):
                a = 1


        return drt


    def grad(self, v_data, rt_p_data, rt_m_data, h_model, v_model, mean_vtm, mean_r_p_model, mean_r_m_model, CDk):
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
        param = [param, param, param, param]# dtheta_p, dtheta_m, dgamma_p, dgamma_m

        drtp_drtp_min_1 = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drp_dU_p = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drp_dW = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drp_db_init_p = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)
        drp_dtheta_p = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drp_dgamma_p = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)

        drtm_drtm_min_1 = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drm_dU_m = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drm_dW = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drm_db_init_m = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)
        drm_dtheta_m = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)
        drm_dgamma_m = torch.zeros(self.N_H, self.T, dtype=self.dtype, device=self.device)


        for t in range(self.T):
            if t==0:
                I_data = torch.matmul(v_data[:, t], self.W.T)
                I_model = torch.matmul(v_model[:, t, -1], self.W.T)
                param_t = self.grad_dReLU(I_data, I_model)
                for i in range(len(param_t)): param[i] = param_t[i]
                dr_db_init_p = self.drt(I_data, derivative='b_init_p')
                dr_db_init_m = self.drt(I_data, derivative='b_init_m')

            elif t>0:
                I_data = torch.matmul(v_data[:, t], self.W.T)
                I_model = torch.matmul(v_model[:, t, -1], self.W.T)

                param_t = self.grad_dReLU(I_data, I_model, rt_p_data[:, t-1], rt_m_data[:, t-1], mean_r_p_model[:, t-1], mean_r_m_model[:, t-1])

                for i in range(len(param_t)): param[i] += param_t[i]

            drp_dtheta_p[:, t] = self.drt(I_data, derivative='theta_p')
            drp_dgamma_p[:, t] = self.drt(I_data, derivative='gamma_p')

            drm_dtheta_m[:, t] = self.drt(I_data, derivative='theta_m')
            drm_dgamma_m[:, t] = self.drt(I_data, derivative='gamma_m')

            drtp_drtp_min_1[:, t] = self.drt(I_data, derivative='rt')
            drtm_drtm_min_1[:, t] = self.drt(I_data, derivative='rt')

            drp_dU_p[:, t] = self.drt(I_data, derivative='U_p')
            drm_dU_m[:, t] = self.drt(I_data, derivative='U_m')
            drp_dW[:, t] = self.drt(I_data, derivative='Wp')
            drm_dW[:, t] = self.drt(I_data, derivative='Wm')

        dtheta_p, dtheta_m, dgamma_p, dgamma_m = param

        # Backpropagation
        Dt_p = torch.zeros(self.N_H, self.T + 1, dtype=self.dtype, device=self.device)
        Dt_m = torch.zeros(self.N_H, self.T + 1, dtype=self.dtype, device=self.device)

        for t in range(self.T - 1, -1, -1): # begin, stop, step
            Dt_p[:, t] = torch.matmul(self.U_p.T, (Dt_p[:, t + 1] * drtp_drtp_min_1[:, t] + (rt_p_data[:, t] - mean_r_p_model[:, t]) ))
            Dt_m[:, t] = torch.matmul(self.U_m.T, (Dt_m[:, t + 1] * drtm_drtm_min_1[:, t] + (rt_m_data[:, t] - mean_r_m_model[:, t]) ))

        # Gradients for HU potential
        dtheta_p += torch.sum(Dt_p[:, 2:self.T] * drp_dtheta_p[:, 1:self.T - 1])
        dgamma_p += torch.sum(Dt_p[:, 2:self.T] * drp_dgamma_p[:, 1:self.T - 1])

        dtheta_m += torch.sum(Dt_m[:, 2:self.T] * drm_dtheta_m[:, 1:self.T - 1])
        dgamma_m += torch.sum(Dt_m[:, 2:self.T] * drm_dgamma_m[:, 1:self.T - 1])

        # Gradient initial bias
        db_init_p = (rt_p_data[:, 0] - mean_r_p_model[:, 0]) + Dt_p[:, 1] * drp_db_init_p
        db_init_m = (rt_m_data[:, 0] - mean_r_m_model[:, 0]) + Dt_m[:, 1] * drp_db_init_m

        # Gradient for visible field
        db_V = torch.sum(v_data - mean_vtm, 1)

        # Gradient for W
        dW_1 = torch.sum(
            (Dt_p[:, 1:self.T] * drp_dW[:, 0:self.T-1]).unsqueeze(1).repeat(1, self.N_V, 1) *
            v_data[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)

        dW_2 = torch.sum(
            (Dt_m[:, 1:self.T] * drm_dW[:, 0:self.T - 1]).unsqueeze(1).repeat(1, self.N_V, 1) *
            v_data[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)

        dW_3 = torch.sum(rtd.unsqueeze(1).repeat(1, self.N_V, 1) * v_data.unsqueeze(0).repeat(self.N_H, 1, 1), 2) - \
               torch.sum(
                   torch.sum(h_model.unsqueeze(1).repeat(1, self.N_V, 1, 1) * v_model.unsqueeze(0).repeat(self.N_H, 1, 1, 1),
                             3), 2) / CDk
        dW = dW_1 + dW_2

        # Gradient for U
        dU_p = torch.sum((Dt_p[:, 2:self.T + 1] * (drp_dU_p_dU[:, 1:self.T]) + rt_p_data[:, 1:self.T] - mean_r_p_model[:, 1:self.T]).unsqueeze(
            1).repeat(1, self.N_H, 1) * rt_p_data[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)

        if torch.sum(torch.isnan(dtheta_p)):
            a = 1
        if torch.sum(torch.isnan(dtheta_m)):
            a = 1
        if torch.sum(torch.isnan(dgamma_p)):
            a = 1
        if torch.sum(torch.isnan(dgamma_m)):
            a = 1


        if torch.sum(torch.isnan(Dt)):
            a=1
        if torch.sum(torch.isnan(db_V)):
            a=1
        if torch.sum(torch.isnan(dW)):
            a=1
        if torch.sum(torch.isnan(dU)):
            a=1

        del Dt

        return [dW, dU, db_V, db_init, dtheta_p, dtheta_m, dgamma_p, dgamma_m]

    def update_grad(self, Dparams, lr=1e-3, mom=0,wc=0, x=2, sp=None):

        dW, dU, db_V, db_init, dtheta_p, dtheta_m, dgamma_p, dgamma_m = self.dparams
        DW, DU, Db_V, Db_init, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Db_V = mom * Db_V + lr * db_V
        Db_init = mom * Db_init + lr * db_init

        Dtheta_p = mom * Dtheta_p + lr * dtheta_p
        Dtheta_m = mom * Dtheta_m + lr * dtheta_m
        Dgamma_p = mom * Dgamma_p + lr * dgamma_p
        Dgamma_m = mom * Dgamma_m + lr * dgamma_m

        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V), \
                                     [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, DU, Db_V, Db_init, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m]

        for i in range(len(self.params)): self.params[i] += Dparams[i]

        return [DW, DU, Db_V, Db_init, Dtheta_p, Dtheta_m, Dgamma_p, Dgamma_m]

    def constraint(self, threshold=1e-2):
        for p in [self.gamma_p, self.gamma_m]:
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

