import torch


class RTRBM(torch.nn.Module):
    def __init__(self, data, n_hidden, device='cpu'):
        super().__init__()
        if data.dim() == 3:
            self.n_visible, self.T, self.n_batches = data.shape
            self.data = data
        else:
            self.n_visible, self.T = data.shape
            self.n_batches = 1
            self.data = data.unsqueeze(2)
        self.n_hidden = n_hidden
        self.device = device

        self.W = torch.nn.Parameter(.005 * torch.randn(self.n_visible, self.n_hidden))
        self.U = torch.nn.Parameter(.005 * torch.randn(self.n_hidden, self.n_hidden))
        self.b_v = torch.nn.Parameter(.005 * torch.randn(self.n_visible))
        self.b_h = torch.nn.Parameter(.005 * torch.randn(self.n_hidden))
        self.b_init = torch.nn.Parameter(.005 * torch.randn(self.n_hidden))

    def learn(self, n_epochs=1000, lr=0.001, momentum=0.9, k=10):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        for epoch in range(n_epochs):
            for batch in range(self.n_batches):
                v = self.data[..., batch].to(self.device)
                # vsk, hsk = [], []

                v_samples = [v]

                # hier 2 opties: r gebruiken of h?? is er uberhaupt een verschil?
                for kk in range(k):

                    h = [torch.sigmoid(torch.nn.functional.linear(
                        input=v_samples[-1][:, 0],
                        weight=self.W.T,
                        bias=self.b_init))]
                    h_sample = torch.bernoulli(h[-1])
                    v_sample = [torch.bernoulli(
                        torch.sigmoid(
                            torch.nn.functional.linear(
                                input=h_sample,
                                weight=self.W,
                                bias=self.b_v)))]

                    # kan linear uitgevoerd worden, maar voor te testen is dit beter
                    for t in range(1, self.T):
                        h += [torch.sigmoid(
                            torch.nn.functional.linear(
                                input=v_samples[-1][:, t],
                                weight=self.W.T,
                                bias=self.b_h)
                            + torch.matmul(self.U, h[-1]))]
                        h_sample = torch.bernoulli(h[-1])
                        v_sample += [torch.bernoulli(
                            torch.sigmoid(
                                torch.nn.functional.linear(
                                    input=h_sample,
                                    weight=self.W,
                                    bias=self.b_v)))]

                    v_samples += [torch.cat(v_sample, dim=1)]

                    if kk == 0:
                        h_data = torch.cat(h, 1)
                    # vsk += [v_samples]
                    # hsk += [h]
                v_model = torch.cat(v_samples, 1)
                h_model = torch.cat(h, 1)
                optimizer.zero_grad()
                # welke h hier gebruiken? mean? bernoulli?
                loss = self.free_energy(v, h_data) - self.free_energy(v_model, h_model)
                loss.backward()
                optimizer.step()

    def calculate_r(self, v):
        T = v.shape[1]

        r = [torch.sigmoid(torch.nn.functional.linear(input=v[:, 0], weigth=self.W.T, bias=self.b_init))]

        for t in range(1, T):
            r += [torch.sigmoid(
                torch.nn.functional.linear(
                    input=v[:, t],
                    weight=self.W.T,
                    bias=self.b_h)
                + torch.matmul(self.U, r[-1]))]
            return torch.cat(r, 1)

    def free_energy(self, v, h):
        T = v.shape[1]

        r = self.calculate_r(v)

        free_energy = [
            torch.matmul(torch.matmul(v[:, 0], self.W), h[:, 0]) +
            torch.dot(v[:, 0], self.b_v) +
            torch.dot(h[:, 0], self.b_init)]

        for t in range(1, T):
            free_energy += [
                torch.matmul(torch.matmul(v[:, t], self.W), h[:, t]) +
                torch.dot(v[:, t], self.b_v) +
                torch.dot(h[:, t], self.b_h) +
                torch.matmul(torch.matmul(r[t - 1], self.U), h[:, t])]

        return sum(free_energy)


if __name__ == '__main__':
    from data.mock_data import create_BB
    data = create_BB()
    rtrbm = RTRBM(
        data=data,
        n_hidden=10
    )
    rtrbm.learn(
        n_epochs=100,
        lr=1e-3
    )

    """
    def visible_to_hidden(self, v):
        h = [torch.sigmoid(torch.nn.functional.linear(
            input=v[:, 0].T,
            weigth=self.W.T,
            bias=self.b_init
        ))]
        for t in range(self.T):
            h += [torch.sigmoid(torch.nn.functional.linear(
                input=v[:, t].T,
                weigth=self.W.T,
                bias=self.b_h
            ))]

        # ht = torch.sigmoid(torch.nn.functional.linear(
        #     input=v[:, 1:].T,
        #     weigth=self.W.T,
        #     bias=self.b_h
        # ))
    """



