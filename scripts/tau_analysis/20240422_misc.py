import numpy as np 

import matplotlib.pyplot as plt
from matplotlib import cm

from common import plot_init_big, PlotSaver

trace_dir = "/mnt/ltio/parthenon-topo/stochsg.56.hybrid90"
evt0_mat_fpath = f"{trace_dir}/evt0.mat.npy"
evt1_mat_fpath = f"{trace_dir}/evt1.mat.npy"

evt0_mat = np.load(evt0_mat_fpath)
evt1_mat = np.load(evt1_mat_fpath)

evt0_mat
evt1_mat

def print_ts_hist(mat, idx):
    ts = mat[idx]
    ts = ts[~np.isnan(ts)]
    ts_min = np.min(ts)
    ts_max = np.max(ts)
    bins = np.arange(ts_min, ts_max, 1000)
    hist, bin_edges = np.histogram(ts, bins=bins)
    print(idx, ts_min, ts_max, ts.shape)
    print(hist)
    # print(ts[:100])

    # idxes = np.where(ts > 50000)
    # print(idxes)
    return ts

def plot_mat_range(mat, start, end):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # colors = cm.viridis(np.linspace(0, 1, end - start + 1))
    # npoints = len(mat[start])
    # colors = cm.viridis(np.linspace(0, 1, npoints))

    point_styles = ['o', 's', '^', '>', '<', 'v', 'p', '*', '+']

    for idx, row in enumerate(mat[start:end]):
        row = row[~np.isnan(row)]
        npoints = len(row)
        colors = cm.viridis(np.linspace(0, 1, npoints))
        print(row)
        data_x = np.arange(len(row))
        # ax.plot(data_x, row, 'o', color=colors[idx])
        p = point_styles[idx % len(point_styles)]
        ax.scatter(data_x, row, marker=p, color=colors)

    ax.set_xlabel("Index")
    ax.set_ylabel("Time (us)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Block Times {start}-{end}")
    ax.grid()
    fig.tight_layout()

    # ax.set_xlim(left=1000, right=1200)

    plt_fname = f"block_times_{start}_{end}"
    PlotSaver.save(fig, "", None, plt_fname)

def plot_corr(mat, idx1, idx2):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ts1 = mat[idx1]
    ts2 = mat[idx2]

    ts1 = ts1[~np.isnan(ts1)]
    ts2 = ts2[~np.isnan(ts2)]

    npoints = len(ts1)
    colors = cm.viridis(np.linspace(0, 1, npoints))

    ax.scatter(ts1, ts2, marker='o', color=colors)

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Time (us)")
    ax.set_title(f"Block Times {idx1} vs {idx2}")
    ax.grid()
    fig.tight_layout()

    plt_fname = f"block_corr_{idx1}_{idx2}"
    PlotSaver.save(fig, "", None, plt_fname)

ts1 = print_ts_hist(evt0_mat, 2000)
ts2 = print_ts_hist(evt0_mat, 2001)
ts3 = print_ts_hist(evt0_mat, 2002)
ts3 = print_ts_hist(evt0_mat, 2010)
ts3 = print_ts_hist(evt0_mat, 2011)

i = 790
for ts in range(2000, 2011):
    print(ts, evt0_mat[ts][i])

plot_init_big()
plot_mat_range(evt0_mat[:, 1000:1200], 2000, 2002)
plot_corr(evt0_mat, 2000, 2001)

evt0_mat[2000:2001]


evt0_mat.nansum(axis=1) / 1e6
mm = np.nansum(evt0_mat, axis=1) / 1e6
mm.astype(int)

def print_row_min(mat, ridx):
    row = mat[ridx]
    row = row[~np.isnan(row)]
    print(ridx, np.min(row))

for row in range(0, 10):
    print_row_min(evt0_mat, row)

