import numpy as np
import torch


def phi(x):
    return torch.exp(x**2 / 2) * (1 - torch.erf(x / 2**.5)) * (torch.pi / 2)**.5

def d_phi(x):
    return x * phi(x) - 1


class ReLuRTRBM(object):
    def __init__(self, n_v, n_h, t):
        self.W = .01 * torch.randn(n_h, n_v)
        self.U = .01 * torch.randn(n_h, n_h)
        self.b_v = torch.zeros(1, n_v)
        self.b_h = torch.zeros(1, n_h)
        self.b_init = torch.zeros(1, n_h)
        self.gamma = torch.ones(1, n_h)
        self.r = torch.empty(n_h, t)

        self.n_v = n_v
        self.n_h = n_h
        self.T = t

    def learn(self, data, n_epochs=1000, lr=1e-3, k=10):
        self.errors = torch.zeros(n_epochs)
        for epoch in range(n_epochs):
            err = 0
            for i in range(data.shape[2]):
                v = data[..., i].clone().detach()
                self.calculate_expected_hidden(v)
                vs, hs = self.CDk(v, k)

                # data statistics
                dWd, dQdW, dUd, db_vd, db_hd, db_initd, dgammad = self.calculate_gradients(v, self.r)

                # model statistics
                dWm, _, dUm, db_vm, db_hm, db_initm, dgammam = self.calculate_gradients(vs, hs)

                # update parameters
                self.W = lr * (dWd - dWm + dQdW)
                self.U = lr * (dUd - dUm)
                self.b_v = lr * (db_vd - db_vm)
                self.b_h = lr * (db_hd - db_hm)
                self.b_init = lr * (db_initd - db_initm)
                self.gamma = lr * (dgammad - dgammam)

                self.constraint()

                err += torch.mean((v - vs)**2)
            self.errors[epoch] = err
            print(err)

    def calculate_expected_hidden(self, v):
        self.r[:, 0] = (torch.matmul(self.W, v[:, 0]) + self.b_init) / self.gamma + \
            1 / (torch.sqrt(self.gamma) * phi((- torch.matmul(self.W, v[:, 0]) - self.b_init) / torch.sqrt(self.gamma)))
        for t in range(1, self.T):
            temp = torch.matmul(self.W, v[:, t]) + self.b_h + torch.matmul(self.U, self.r[:, t-1])
            self.r[:, t] = temp / self.gamma + 1 / (torch.sqrt(self.gamma) * phi(- temp / torch.sqrt(self.gamma)))
        return

    def visible_to_hidden(self, v):
        mean = torch.concat([(self.b_init * self.r[:, 0] + self.b_h + torch.matmul(self.W, v[:, 0])).T / self.gamma.T,
                    (torch.matmul(self.U, self.r[:, 1:]) + self.b_h.T + torch.matmul(self.W, v[:, 1:])) / self.gamma.T], 1)
        std = 1 / self.gamma.T
        return max(torch.zeros(self.n_h, self.T), std * (torch.fmod(torch.randn(self.n_h, self.T), 2)) + mean)

    def hidden_to_visible(self, h):
        return torch.bernoulli(torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v.T))

    def CDk(self, v_d, k):
        h = torch.zeros(self.n_h, k+1)
        v = torch.zeros(self.n_v, k+1)

        h[:, 0] = self.visible_to_hidden(v_d)
        v[:, 0] = v_d.detach().clone()

        for kk in range(1, k):
            v[:, kk+1] = self.hidden_to_visible(h[:, kk])
            h[:, kk+1] = self.visible_to_hidden(v[:, kk])

        return torch.mean(v, 1), torch.mean(h, 1)

    def calculate_gradients(self, v, h):
        dQdb_h, dQdU, dQdW, dQdgamma, dQtp1_drt = 0, 0, 0, 0, 0

        for t in range(self.T-2, -1, -1):
            dQtdr = self.dQtp1_drt(v, h, t, dQtp1_drt)
            drtdb_h = self.drt_db_h(t, v)
            drtdU = self.drt_dU(t, v)
            drtdW = self.drt_dW(t, v)
            drtdgamma = self.drt_dgamma(t, v)

            dQdb_h += dQtdr * drtdb_h
            dQdU += torch.outer(dQtdr * drtdU, self.r[:, t - 1]) # add chainrule
            dQdW += torch.outer(dQtdr * drtdW, v[:, t]) # add chainrule
            dQdgamma += dQtdr * drtdgamma

        dHdW = torch.sum(h[:, None, :].repeat(1, self.n_v, 1) * v[None, :, :].repeat(self.n_h, 1, 1), 2) # torch.outer(h, v) but sum over time dimension
        dHb_v = torch.sum(v, 1)
        dHb_h = torch.sum(h[:, 1:], 1)
        dHdb_init = h[:, 0]

        return dHdW, dQdW, dQdU, dHb_v[None, :], dHb_h[None, :] + dQdb_h[None, :], dHdb_init[None, :], dQdgamma[None, :]

    ###########################
    # All partial derivatives #
    ###########################

    def dQtp1_drt(self, v, h, t, dQtp1_drt):
        dQtp1_drt = torch.matmul(self.U.T, dQtp1_drt * self.drtp1_drt(t, v).T + h[:, t])
        return dQtp1_drt

    def drtp1_drt(self, t, v):
        I = - (self.b_h + torch.matmul(self.U, self.r[:, t - 1]) + torch.matmul(self.W, v[:, t])) / torch.sqrt(self.gamma)
        return (1 / self.gamma + d_phi(I) / (self.gamma * phi(I)**2))[0]

    def drt_db_h(self, t, v):
        return self.drtp1_drt(t, v)

    def drt_dU(self, t, v):
        return self.drtp1_drt(t, v)

    def drt_dW(self, t, v):
        return self.drtp1_drt(t, v)

    def drt_dgamma(self, t, v):
        y = - (self.b_h + torch.matmul(self.U, self.r[:, t - 1]) + torch.matmul(self.W, v[:, t])) / torch.sqrt(self.gamma)
        drt_dgamma = (y - 1 / (2 * phi(y)) + y * d_phi(y) / (2 * phi(y)**2)) / self.gamma**(3/2)
        return drt_dgamma[0]

    def constraint(self, threshold=1e-3):
        for p in [self.gamma]:
            idx = p < threshold
            p[idx] = threshold

if __name__ == '__main__':
    from data.mock_data import create_BB
    import seaborn as sns

    data = create_BB(N_V=20, T=32, n_samples=100)
    sns.heatmap(data[..., 0])
    rtrbm = ReLuRTRBM(20, 10, 32)
    rtrbm.learn(data, n_epochs=100, lr=1e-4, k=10)

