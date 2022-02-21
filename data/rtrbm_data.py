import numpy as np
import torch
from tqdm import tqdm
from boltzmann_machines.RTRBM_ import RTRBM


def get_rtrbm_data(N_H=3,
                   N_V=20,
                   T=30,
                   n_batches=300,
                   device='cuda',
                   pre_gibbs_k=100,
                   gibbs_k=100,
                   mode=1):

    # TODO: add sparsity
    W = 2 * np.random.rand(N_H, N_V) - 1
    U = 2 * np.random.rand(N_H, N_H) - 1
    data = torch.zeros(N_V, T, dtype=torch.float)
    rtrbm = RTRBM(data, N_H=N_H, device=device)
    rtrbm.W = torch.tensor(W, device=device, dtype=torch.float)
    rtrbm.U = torch.tensor(U, device=device, dtype=torch.float)

    data = torch.zeros(N_V, T, n_batches)
    rt = torch.zeros(N_H, T, n_batches)
    for batch in tqdm(range(n_batches)):
        v_start = (torch.rand(N_V) > 0.2) * 1.0
        data[:, :, batch], rt[:, :, batch] = rtrbm.sample(v_start.type(torch.float),
                                                          chain=T,
                                                          pre_gibbs_k=pre_gibbs_k,
                                                          gibbs_k=gibbs_k,
                                                          mode=mode,
                                                          disable_tqdm=True)

    return rtrbm, data
