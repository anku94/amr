import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from common import plot_init, PlotSaver
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from trace_reader import TraceReader, TraceOps


def compute_straggler_savings(trace_dir):
    tr = TraceOps(trace_dir)
    ag_mat = tr.multimat("tau:AR3_UMBT")
    ag_mat.sort(axis=1)
    # all_time_spent has 512 rows. each row has 30k cols
    # each row is the time spent by the ith most straggling rank on UMBT
    # this time is negative
    # this is time saved by preventing that rank from straggling

    #  mean_time_spent = np.mean(ag_mat, axis=1)
    #  all_time_spent = ag_mat.T - mean_time_spent

    ts_rank_savings = np.diff(ag_mat)
    rank_ts_savings = ts_rank_savings.T
    rank_ts_savings = np.cumsum(rank_ts_savings, axis=1)
    rank_savings = np.cumsum(rank_ts_savings[:, -1])

    return rank_ts_savings, rank_savings


def plot_straggler_savings_aggr(rank_savings, trace_dir):
    nitems = 100
    data_x = list(range(100))
    data_y = rank_savings[:100]

    fig, ax = plt.subplots(1, 1)

    ax.plot(data_x, data_y)
    ax.set_xlabel("Stragglers fixed (out of 512)")
    ax.set_ylabel("Time saved (s)")
    trpref = trace_dir.split("/")[-1]
    ax.set_title(f"Impact of Straggler Mitigation on Total Runtime ({trpref})")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=4))

    ax.yaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='minor', color='#ddd')
    ax.xaxis.grid(visible=True, which='major', color='#aaa')

    fig.tight_layout()

    PlotSaver.save(fig, trace_dir, None, "straggler_impact_aggr")


def plot_straggler_savings_tswise(rank_ts_savings, trace_dir):
    ranks_to_plot = [0, 1, 2, 5, 10, 20, 50]

    data_x = list(range(rank_ts_savings.shape[1]))

    fig, ax = plt.subplots(1, 1)
    for rank in ranks_to_plot:
        label = f"Rank {rank}"
        data_y = rank_ts_savings[rank]
        ax.plot(data_x, data_y, label=label)

    ax.legend()
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Cumulative Time Saved (s)")
    trpref = trace_dir.split("/")[-1]
    ax.set_title(f"Impact of Straggler Mitigation on Total Runtime ({trpref})")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=4))

    ax.yaxis.grid(visible=True, which='major', color='#aaa')
    ax.yaxis.grid(visible=True, which='minor', color='#ddd')
    ax.xaxis.grid(visible=True, which='major', color='#aaa')

    fig.tight_layout()

    PlotSaver.save(fig, trace_dir, None, "straggler_impact_tswise")


def plot_straggler_savings(trace_dir):
    rank_ts_savings, rank_savings = compute_straggler_savings(trace_dir)
    plot_straggler_savings_aggr(rank_savings, trace_dir)
    plot_straggler_savings_tswise(rank_ts_savings, trace_dir)
    pass


def run():
    plot_init()

    trace_dir = "/mnt/ltio/parthenon-topo/profile10"
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    trace_dir = "/mnt/ltio/parthenon-topo/profile11"
    print(f"Plotting trace: {trace_dir}")
    plot_straggler_savings(trace_dir)


if __name__ == "__main__":
    run()
