import torch
import numpy as np
from tqdm import tqdm
from data.reshape_data import reshape_from_batches, train_test_split
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib import cm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from data.load_data import load_data


def infer_and_get_moments_plot(dir, test, pre_gibbs_k=0, gibbs_k=1, mode=1, n=1000, m=50000):
    rtrbm = torch.load(dir, map_location='cpu')

    rtrbm.device = 'cpu'

    vt = test.clone().detach()
    ht = torch.empty(rtrbm.N_H, rtrbm.T, test.shape[2])
    vs = torch.empty(rtrbm.N_V, rtrbm.T, test.shape[2])
    hs = torch.empty(rtrbm.N_H, rtrbm.T, test.shape[2])

    for i in tqdm(range(test.shape[2])):
        rt = rtrbm.visible_to_expected_hidden(test[:, :, i])
        x, _ = rtrbm.visible_to_hidden(test[:, :, i], rt)
        ht[:, :, i] = x.clone().detach()
        v, h = rtrbm.sample(test[:, 0, i], chain=test.shape[1], pre_gibbs_k=pre_gibbs_k,
                            gibbs_k=gibbs_k, mode=mode, disable_tqdm=True)
        vs[:, :, i] = v.clone().detach().cpu()
        hs[:, :, i] = h.clone().detach().cpu()

    vt = reshape_from_batches(vt)
    ht = reshape_from_batches(ht)
    vs = reshape_from_batches(vs)
    hs = reshape_from_batches(hs)

    vvt, vvs, vht, vhs, hht, hhs = calculate_moments(vt, ht, vs, hs, n=n, m=m)

    vt_mean = np.mean(np.array(vt), axis=1)
    vs_mean = np.mean(np.array(vs), axis=1)
    ht_mean = np.mean(np.array(ht), axis=1)
    hs_mean = np.mean(np.array(hs), axis=1)

    ax = density_plot_moments(vt_mean, vs_mean, ht_mean, hs_mean, vvt, vvs, hht, hhs, vht, vhs)
    plt.show()
    return vt, vs, ht, hs


def calculate_moments(vt, ht, vs, hs, n=1000, m=50000):
    if vt.shape[0] > n:
        idx = torch.randperm(vt.shape[0])[:n]
        vt = vt[idx, :]
        vs = vs[idx, :]

    vvt = np.array(torch.matmul(vt, vt.T) / vt.shape[1]).flatten()
    vvs = np.array(torch.matmul(vs, vs.T) / vs.shape[1]).flatten()
    vht = np.array(torch.matmul(vt, ht.T) / vt.shape[1]).flatten()
    vhs = np.array(torch.matmul(vs, hs.T) / vs.shape[1]).flatten()
    hht = np.array(torch.matmul(ht, ht.T) / ht.shape[1]).flatten()
    hhs = np.array(torch.matmul(hs, hs.T) / hs.shape[1]).flatten()

    if vvt.shape[0] > m:
        idx = torch.randperm(vvt.shape[0])[:m]
        vvt = vvt[idx]
        vvs = vvs[idx]
    if vht.shape[0] > m:
        idx = torch.randperm(vht.shape[0])[:m]
        vht = vht[idx]
        vhs = vhs[idx]
    if hht.shape[0] > m:
        idx = torch.randperm(hht.shape[0])[:m]
        hht = vht[idx]
        hhs = vhs[idx]

    return vvt, vvs, vht, vhs, hht, hhs


def density_scatter(x, y, ax=None, fig=None, r=None, sort=True, bins=20, **kwargs):
    if ax is None:
        fig , ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=2, **kwargs )

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    ax.plot([0, 1], [0, 1], ':')
    ax.set_xlim([0, np.max(x)])
    ax.set_ylim([0, np.max(y)])
    ax.set_xticks([0, np.floor(10 * np.max(x)) / 10])
    ax.set_yticks([0, np.floor(10 * np.max(y)) / 10])

    if r is None:
        r, _ = pearsonr(x, y)
    ax.text(.1, .9, 'r-value: {:.2f}'.format(r), transform=ax.transAxes)

    return ax


def density_plot_moments(vt_mean, vs_mean, ht_mean, hs_mean, vvt, vvs, hht, hhs, vht, vhs):
    fig, axes = plt.subplots(1, 5, figsize=(30, 4))
    density_scatter(vt_mean, vs_mean, ax=axes[0], fig=fig)
    density_scatter(ht_mean, hs_mean, ax=axes[1], fig=fig)
    density_scatter(vvt, vvs, ax=axes[2], fig=fig)
    density_scatter(hht, hhs, ax=axes[3], fig=fig)
    density_scatter(vht, vhs, ax=axes[4], fig=fig)
    return axes


if __name__ == '__main__':
    spikes, behavior, coordinates, df, stimulus = load_data(
        '/mnt/data/zebrafish/chen2018/subject_1/Deconvolved/subject_1_reconv_spikes.h5')
    # sort spikes by ascending firing rate
    firing_rates = np.mean(spikes, 1)
    sort_idx = np.argsort(firing_rates)[::-1]
    firing_rates_sorted = firing_rates[sort_idx]
    data = spikes[sort_idx, :] > .15
    data = torch.tensor(data, dtype=torch.float)

    # split in 80 train batches and 20 test batches
    train, test = train_test_split(data[:50000, :], train_batches=80, test_batches=20)
    vt, vs, ht, hs = infer_and_get_moments_plot('../data/full brain/full_brain_wrong.pt', test)
