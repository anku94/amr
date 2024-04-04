import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import subprocess
import sys
import ray
import traceback
from typing import Tuple, Dict

from common import label_map
from trace_reader import TraceOps


def row_to_idx_ktiles(row, k):
    r_widx = enumerate(row)
    r_sorted_idx = list(map(lambda x: x[0], sorted(r_widx, key=lambda x: x[1])))
    r_ksplits = np.array_split(r_sorted_idx, k)
    r_ksplit_set = [set(r) for r in r_ksplits]
    return r_ksplit_set


def compute_olap_pct(s1, s2):
    try:
        assert len(s1) == len(s2)
    except AssertionError as e:
        print("ERROR: {} != {}".format(len(s1), len(s2)))
        return

    scomm = s1.intersection(s2)
    return len(scomm) * 1.0 / len(s1)


def compute_olap_series(s):
    olap_scores = []
    for idx in range(1, len(s)):
        prev = s[idx - 1]
        cur = s[idx]
        olappct = compute_olap_pct(prev, cur)
        olap_scores.append(olappct)

    return olap_scores


def compute_recurrence_mat(mat, k):
    matl = mat.tolist()
    matl_rsplits = map(lambda x: row_to_idx_ktiles(x, k), matl)

    # [ts][ktile] = {set} to
    # [ktile][ts] = {set}
    ktile_data = list(zip(*matl_rsplits))
    ktile_olaps = list(map(compute_olap_series, ktile_data))

    return ktile_olaps


def plot_ktile_all(k, all_ktiles, mat_label, plotdir, save=False):
    print(k, mat_label, plotdir)

    mat_label_fs = mat_label.replace(":", "_")
    plot_name = "{}.k{}.allk.pdf".format(mat_label_fs, k)
    plot_fullpath = "{}/{}".format(plotdir, plot_name)
    print("Generating {}".format(plot_fullpath))

    fig, ax = plt.subplots(1, 1)

    data_x = range(len(all_ktiles[0]))

    for ktidx, ktile in enumerate(all_ktiles):
        label = "K-Tile {}".format(ktidx)
        print(label)
        ax.plot(data_x, ktile, label=label, alpha=0.8)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sim-Pct Vs Prev TS")
    ax.set_title("All K-Tiles (Evt {}, K={})".format(label_map[mat_label], k))

    ax.set_ylim([0, 1])
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}%".format(x * 100))

    if (k <= 8)
        ax.legend()

    if save:
        fig.savefig(plot_fullpath, dpi=300)
    else:
        fig.show()


def plot_ktile_top(all_ks, all_top_ktiles, mat_label, plotdir, save=False):
    mat_label_fs = mat_label.replace(":", "_")
    num_ks = len(all_ks)
    plot_name = "{}.topktiles.n{}.pdf".format(mat_label_fs, num_ks)
    plot_fullpath = "{}/{}".format(plotdir, plot_name)
    print("Generating {}".format(plot_fullpath))

    fig, ax = plt.subplots(1, 1)

    data_x = range(len(all_top_ktiles[0]))

    for k, top_ktile in zip(all_ks, all_top_ktiles):
        label = "Top {}%".format(100.0/k)
        print(label)
        ax.plot(data_x, top_ktile, label=label, alpha=0.8)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sim-Pct Vs Prev TS")
    ax.set_title("Top K-Tiles For Evt {}".format(label_map[mat_label]))

    ax.set_ylim([0, 1])
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}%".format(x * 100))

    ax.legend()

    if save:
        fig.savefig(plot_fullpath, dpi=300)
    else:
        fig.show()
    pass


def plot_recurrence_suite(mat, mat_label, plotdir):
    all_k = [4, 8, 16, 32]
    all_k_olaps = list(map(lambda k: compute_recurrence_mat(mat, k), all_k))
    all_top_olaps = list(map(lambda x: x[-1], all_k_olaps))


    #  for k, k_olap in zip(all_k, all_k_olaps):
        #  plot_ktile_all(k, k_olap, mat_label, plotdir, save=True)

    for num_k in [2, 3, 4]:
        all_k_toplot = all_k[:num_k]
        all_top_olaps_toplot = all_top_olaps[:num_k]
        plot_ktile_top(all_k_toplot, all_top_olaps_toplot, mat_label, plotdir, save=True)


def run_recurrence(tracedir: str, plotdir: str) -> None:
    print(tracedir)
    tr = TraceOps(tracedir)

    mat_labels = ["tau:AR1", "tau:SR", "tau:AR2", "tau:AR3-AR3_UMBT"]
    mats = [tr.multimat(l) for l in mat_labels]

    for mat_label, mat in zip(mat_labels, mats):
        plot_recurrence_suite(mat, mat_label, plotdir)
    #  ar1 = tr.multimat("tau:AR1")
    #  sr = tr.multimat("tau:SR")
    #  ar2 = tr.multimat("tau:AR2")
    #  ar3 = tr.multimat("tau:AR3-AR3_UMBT")
    pass


def run():
    tracedir = "/mnt/ltio/parthenon-topo/profile8"
    plotdir = "figures/20220830"
    run_recurrence(tracedir, plotdir)


if __name__ == "__main__":
    run()
