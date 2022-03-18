import numpy as np
import torch


def phi(x):
    return torch.exp(x**2 / 2) * (1 - torch.erf(x / 2**.5)) * (torch.pi / 2)**.5

def d_phi(x):
    return x * phi(x) - 1


class ReLuRTRBM(object):
    def __init__(self, n_v, n_h, t):
        self.W = .01/n_v * torch.randn(n_h, n_v)
        self.U = .01/n_h * torch.randn(n_h, n_h)
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
                dWd, dUd, db_vd, db_hd, db_initd, dgammad = self.calculate_gradients(v, self.r)

                # model statistics
                dWm, dUm, db_vm, db_hm, db_initm, dgammam = self.calculate_gradients(vs, hs)

                # update parameters
                self.W = lr * (dWd - dWm)
                self.U = lr * (dUd - dUm)
                self.b_v = lr * (db_vd - db_vm)
                self.b_h = lr * (db_hd - db_hm)
                self.b_init = lr * (db_initd - db_initm)
                self.gamma = lr * (dgammad - dgammam)

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
        mean = (torch.matmul(self.U, self.r) + self.b_h.T + torch.matmul(self.W, v)) / self.gamma.T
        std = 1 / self.gamma.T
        return std * (torch.fmod(torch.randn(self.n_h, self.T), 2)) + mean

    def hidden_to_visible(self, h):
        return torch.bernoulli(torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v.T))

    def CDk(self, v, k):
        h = torch.zeros(self.n_h, self.T, k)
        v = torch.zeros(self.n_v, self.T, k)
        h = self.visible_to_hidden(v)
        for kk in range(k):
            v[..., kk] = self.hidden_to_visible(h)
            h[..., kk] = self.visible_to_hidden(v)
        return v / (k + 1), h / (k + 1)

    def calculate_gradients(self, v, h):
        dQdb_h, dQdU, dQdW, dQdgamma, dQtdr = 0, 0, 0, 0, 0

        for t in range(self.T, -1, -1):
            dQtdr = self.dQtdr(v, h, t, dQtdr)
            drtdb_h = self.drtdb_h(t, v)
            drtdU = self.drtdU(t, v)
            drtdW = self.drtdW(t, v)
            drtdgamma = self.drtdgamma(t, v)

            dQdb_h += dQtdr * drtdb_h
            dQdU += dQtdr * drtdU
            dQdW += dQtdr * drtdW
            dQdgamma += dQtdr * drtdgamma

        dHdW = torch.outer(v, h)
        dHb_v = torch.sum(v, 1)
        dHb_h = torch.sum(h[:, 1:], 1)
        dHdb_init = h[:, 0]

        return dHdW, dQdU, dHb_v, dHb_h + dQdb_h, dHdb_init, dQdgamma

    ###########################
    # All partial derivatives #
    ###########################

    def dQtdr(self, v, h, t, dQt_min1_dr):
        dQtdr = torch.matmul(self.U.T, dQt_min1_dr * self.drtdb_h(t, v) + h[:, t + 1])
        return dQtdr

    def drtdb_h(self, t, v):
        y = - (self.b_h.T + torch.matmul(self.U, self.r[:, t]) + torch.matmul(self.W, v[:, t])) / torch.sqrt(self.gamma.T)
        return 1 / self.gamma + d_phi(y) / (self.gamma * phi(y)**2)

    def drtdU(self, t, v):
        return self.drtdb_h(t, v) * self.r[:, t]

    def drtdW(self, t, v):
        return self.drtdb_h(t, v) * v

    def drtdgamma(self, t, v):
        y = - (self.b_h + torch.matmul(self.U, self.r[:, t]) + torch.matmul(self.W, v[:, t])) / torch.sqrt(self.gamma)
        drtdgamma = (y - 1 / (2 * phi(y)) + y * d_phi(y) / (2 * phi(y)**2)) / self.gamma**(3/2)
        return drtdgamma


if __name__ == '__main__':
    from data.mock_data import create_BB
    import seaborn as sns

    data = create_BB(N_V=20, T=32, n_samples=100)
    sns.heatmap(data[..., 0])
    rtrbm = ReLuRTRBM(20, 10, 32)
    rtrbm.learn(data, n_epochs=100, lr=1e-3, k=10)

