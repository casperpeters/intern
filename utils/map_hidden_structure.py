import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
from utils.funcs import get_hidden_mean_receptive_fields as rf
import torch


class MapHiddenStructure(object):
    def __init__(self, W=None, U=None, rtrbm=None, dir=None, coordinates=None):
        if W is not None:
            self.W = W
        if U is not None:
            self.U = U
        if dir is not None:
            rtrbm = torch.load(dir, map_location='cpu')
        if rtrbm is not None:
            self.W = rtrbm.W
            self.U = rtrbm.U

        self.cmap = plt.get_cmap('tab20')

        self.n_h, self.n_v = self.W.shape

        if coordinates is None:
            # assume an even distribution of visible neurons over populations
            coordinates = torch.empty(self.n_v, 2)
            self.n_v_pop = self.n_v // self.n_h
            theta = torch.linspace(0, 2 * torch.pi, self.n_h + 1)
            x, y = torch.cos(theta), torch.sin(theta)
            for pop in range(self.n_h):
                coordinates[pop * self.n_v_pop:(pop + 1) * self.n_v_pop, 0] = \
                    x[pop] + .2 * torch.sin(.5 * theta[1]) * torch.randn(self.n_v_pop)
                coordinates[pop * self.n_v_pop:(pop + 1) * self.n_v_pop, 1] = \
                    y[pop] + .2 * torch.sin(.5 * theta[1]) * torch.randn(self.n_v_pop)
        self.coordinates = coordinates

        self.rf = self.receptive_fields()

    def draw_final_structure(self, hidden_weight_threshold=.5, r=.1, vr=2, save=False, path='', ax=None):
        if ax is None:
            ax = plt.subplot()

        theta = torch.deg2rad(torch.tensor(200))
        max_hidden_connection = torch.max(torch.abs(self.W), 0)[1]
        U_norm = self.U / torch.max(self.U)

        for h, (x, y) in enumerate(self.rf):
            # draw visible neurons as dots
            ax.scatter(self.coordinates[max_hidden_connection == h, 0],
                       self.coordinates[max_hidden_connection == h, 1],
                       color=self.cmap.colors[2 * h], s=vr)

            # draw hidden neurons as circles
            circle = plt.Circle((x, y), radius=r, fill=False, color=self.cmap.colors[2 * h])
            ax.add_patch(circle)
            ax.text(x, y - .03, str(h), ha='center', fontsize=10)

            # draw hidden connections as arrows
            for hh in range(self.n_h):
                u = U_norm[h, hh]
                color_ = 'red' if u < 0 else 'green'
                width_ = torch.abs(u)
                if abs(u) > hidden_weight_threshold:

                    # self-connecting arrow
                    if h == hh:
                        arc = Arc((x, y + r), r * 2, r * 2, angle=-30, theta1=0, theta2=230, capstyle='round',
                                  linestyle='-', lw=width_, color=color_)

                        X, Y = x + r * torch.cos(theta), y + r * torch.sin(theta)
                        arc_head = RegularPolygon((X, Y + r), 3, r / 5, theta.item(), color=color_)
                        ax.add_patch(arc_head)
                        ax.add_patch(arc)

                    # draw arrow
                    else:
                        x2, y2 = self.rf[hh]
                        angle = torch.atan2(x2 - x, y2 - y)
                        dx, dy = r * torch.sin(angle), r * torch.cos(angle)
                        ax.arrow(x+dx, y+dy, x2-x-2*dx, y2-y-2*dy, lw=width_, color=color_, length_includes_head=True,
                                 head_width=width_ / 30, overhang=0)
            ax.axis('off')
            ax.axis('square')

            if save:
                plt.savefig(path, dpi=500)
        return ax

    def receptive_fields(self, coordinates=None, weights=None):
        if weights is None:
            weights = self.W
        if coordinates is None:
            coordinates = self.coordinates
        return (torch.matmul(torch.abs(weights), coordinates).T / torch.sum(torch.abs(weights), 1)).T


if __name__ == '__main__':
    n_v, n_h = (100, 10)
    n_v_pop = n_v // n_h
    W = torch.zeros(n_h, n_v)
    U = torch.randn(n_h, n_h)
    for pop in range(n_h):
        W[pop, pop * n_v_pop:(pop + 1) * n_v_pop] = 1

    x = MapHiddenStructure(W=W, U=U)
    plt.figure(dpi=200)
    ax = x.draw_final_structure(ax=plt.gca())
    plt.show()
