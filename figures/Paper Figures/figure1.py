import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')

from boltzmann_machines.cp_rbm import RBM
from boltzmann_machines.cp_rtrbm import RTRBM


def create_machines():

    n_h, n_v, T = 5, 15, 30

    W = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ], dtype=torch.float)

    U = torch.tensor([
        [-1, 1, -1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, -1, 1, -1],
        [-1, -1, -1, -1, 1],
        [1, -1, -1, -1, -1],
    ], dtype=torch.float)

    W[W == 0] = -1
    W *= 3
    U *= 3
    data = torch.zeros(n_v, T, 10)
    rbm = RBM(data, N_H=n_h, device='cpu', debug_mode=True)
    rtrbm = RTRBM(data, N_H=n_h, device='cpu', debug_mode=True)

    rbm.W = W + .5 * torch.randn(n_h, n_v)
    rtrbm.W = W + .5 * torch.randn(n_h, n_v)
    rtrbm.U = U + .5 * torch.randn(n_h, n_h)
    return rbm, rtrbm


if __name__ == '__main__':

    # create data
    rbm, rtrbm = create_machines()
    v = (torch.rand(rbm.N_V) < .5).T.type(torch.float)
    vs_rbm, hs_rbm = rbm.sample(v_start=v, pre_gibbs_k=0, gibbs_k=100, chain=rbm.T)
    vs_rtrbm, hs_rtrbm = rtrbm.sample(v_start=v, pre_gibbs_k=0, gibbs_k=100, chain=rtrbm.T)

    # create figure and grid-spec
    fig = plt.figure(figsize=(6.692913379, 3.149606296))
    gs = fig.add_gridspec(nrows=7, ncols=11, hspace=0.1, wspace=0.1, left=0.02, right=0.98, top=0.95, bottom=0.05)
    fs = 8

    # create all axes (save space for illustrations)
    W_rbm = fig.add_subplot(gs[2, 4:6])
    cbar = fig.add_subplot(gs[2, 6])
    W_rtrbm = fig.add_subplot(gs[4, 4:6])
    U_rtrbm = fig.add_subplot(gs[4, 6])
    h_rbm = fig.add_subplot(gs[0, 8:])
    v_rbm = fig.add_subplot(gs[1:3, 8:])
    h_rtrbm = fig.add_subplot(gs[4, 8:])
    v_rtrbm = fig.add_subplot(gs[5:, 8:])
    axes = [W_rbm, cbar, W_rtrbm, U_rtrbm, h_rbm, v_rbm, h_rtrbm, v_rtrbm]

    # turn off all axes ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # visible-hidden weights rbm
    pc = W_rbm.imshow(rbm.W, cmap=plt.get_cmap('bwr'), aspect='auto')

    # visible-hidden weights rtrbm
    W_rtrbm.imshow(rtrbm.W, cmap=plt.get_cmap('bwr'), aspect='auto')

    # hidden-hidden weights rtrbm
    U_rtrbm.imshow(rtrbm.U, cmap=plt.get_cmap('bwr'), aspect='auto')

    # weights colorbar
    pos = cbar.get_position()
    new_pos = [pos.x0, pos.y0, pos.width, pos.height / 6]
    cbar.set_position(new_pos)
    fig.colorbar(pc, cbar, orientation='horizontal')
    cbar.set_xticks([-2, 2])
    cbar.set_xlabel('Weight', fontsize=fs)
    cbar.xaxis.tick_top()
    cbar.xaxis.set_label_position('top')
    cbar.tick_params(labelsize=fs)
    cbar.spines[:].set_linewidth(.5)

    # activities rbm
    h_rbm.imshow(hs_rbm, cmap=plt.get_cmap('binary'), aspect='auto')
    v_rbm.imshow(vs_rbm, cmap=plt.get_cmap('binary'), aspect='auto')

    # activities rtrbm
    h_rtrbm.imshow(hs_rtrbm, cmap=plt.get_cmap('binary'), aspect='auto')
    v_rtrbm.imshow(vs_rtrbm, cmap=plt.get_cmap('binary'), aspect='auto')

    # save and show fig
    plt.savefig(r'D:\OneDrive\RU\Intern\rtrbm_master\figures\structure.png', dpi=300)
    plt.show()
