import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from common import plot_init, PlotSaver
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from trace_reader import TraceReader, TraceOps


def get_ts_data(trace_path: str):
    tr = TraceReader(trace_path)
    tr_stats = tr.read_logstats()
    data = tr_stats["wtime_step_amr"] + tr_stats["wtime_step_other"]

    return data


"""
Plot a histogram of timestep distributions
For two traces
"""


def plot_ts_hist(trace_a: str, trace_b: str):
    data_a = get_ts_data(trace_a)
    data_b = get_ts_data(trace_b)

    hist_a, bins = np.histogram(data_a, bins=1000)
    hist_b, _ = np.histogram(data_b, bins=bins)
    width = bins[1] - bins[0]

    fig, ax = plt.subplots(1, 1)
    ax.step(bins[:-1], hist_a, where="post")
    ax.step(bins[:-1], hist_b, where="post")

    ax.set_title("Histogram Of Timestep Times")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("# Occurrences")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{:.0f}".format(x)))
    fig.tight_layout()

    PlotSaver.save(fig, None, None, "distrib_compare")


"""
Plot top-k timestep similarity
For two traces
"""


def plot_ts_topk_sim(trace_a: str, trace_b: str):
    times_a = get_ts_data(trace_a)
    times_b = get_ts_data(trace_b)

    get_sorted_ts = lambda times: list(
        map(lambda x: x[0], sorted(list(enumerate(times)), key=lambda x: x[1]))
    )

    def get_sim(sorted_a, sorted_b, top_k):
        topk_a = sorted_a[-top_k:]
        topk_b = sorted_b[-top_k:]
        sim_ab = len(set(topk_a).intersection(set(topk_b)))
        sim_pct = sim_ab * 1.0 / len(topk_a)
        return sim_pct

    sts_a = get_sorted_ts(times_a)
    sts_b = get_sorted_ts(times_b)
    get_sim(sts_a, sts_b, 100)

    k = 30000
    data_x = list(range(k, 0, -1))
    data_y = list(map(lambda x: get_sim(sts_a, sts_b, x), data_x))

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y)

    ax.invert_xaxis()
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}%".format(x * 100))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_title("Similarity of Top-K Timesteps")
    ax.set_ylabel("Similarity (%)")
    ax.set_xlabel("K")

    ax.set_ylim(ymin=0, ymax=1)
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")
    fig.tight_layout()

    PlotSaver.save(fig, None, None, f"topk_sim_{k}")


def get_idxmat(trace_path: str):
    tr = TraceOps(trace_path)
    trdata = tr.multimat("tau:AR3_UMBT")
    idxmat = np.argsort(trdata, axis=1)
    return idxmat


def get_rank_sim(idxmat_a, idxmat_b, k):
    mma = idxmat_a[:, :k]
    mmb = idxmat_b[:, :k]
    common_len = []
    for row_a, row_b in zip(mma, mmb):
        int_ab = set.intersection(set(row_a), set(row_b))
        common_len.append(len(int_ab))

    simpct = np.array(common_len, dtype=float) / k
    return simpct

def plot_rank_topk_sim_internal(idxmat_a, idxmat_b, all_k):
    fig, ax = plt.subplots(1, 1)

    nts = idxmat_a.shape[0]
    data_x = list(range(nts))

    for k in all_k:
        simpct_k = get_rank_sim(idxmat_a, idxmat_b, k)
        ax.plot(data_x, simpct_k, label=f"K={k}")

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}%".format(x * 100))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_ylim(ymin=0, ymax=1)
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")
    ax.legend()

    ax.set_title("Top-K Rank Similarity for Two Traces")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Similarity (%)")

    fig.tight_layout()

    klabel = ''.join([str(i) for i in all_k])
    PlotSaver.save(fig, None, None, f"topk_sim_rank_{klabel}")


def plot_rank_topk_sim(trace_a, trace_b):
    idxmat_a = get_idxmat(trace_a)
    idxmat_b = get_idxmat(trace_b)

    plot_rank_topk_sim_internal(idxmat_a, idxmat_b, [10])
    plot_rank_topk_sim_internal(idxmat_a, idxmat_b, [20])
    plot_rank_topk_sim_internal(idxmat_a, idxmat_b, [50])
    plot_rank_topk_sim_internal(idxmat_a, idxmat_b, [100])


def run():
    plot_init()

    trace_a = "/mnt/ltio/parthenon-topo/profile10"
    trace_b = "/mnt/ltio/parthenon-topo/profile11"
    print(f"Plotting traces: \n{trace_a}\n{trace_b}")
    #  plot_ts_hist(trace_a, trace_b)
    #  plot_ts_topk_sim(trace_a, trace_b)
    plot_rank_topk_sim(trace_a, trace_b)


if __name__ == "__main__":
    run()
