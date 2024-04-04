import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import re

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analyze_msgs import aggr_msgs_all, join_msgs, get_relevant_pprof_data

from common import plot_init_big as plot_init, PlotSaver

global tdir


def read_perf_df(rank: int) -> pd.DataFrame:
    global tdir
    fpath = f"{tdir}/profile/data.{rank}.csv"
    print(fpath)
    #  fpath = "/mnt/ltio/parthenon-topo/athenapk11/profile/data.{}.csv".format(rank)
    df = pd.read_csv(fpath, names=["usec"])
    df["rank"] = rank
    return df


def read_perf_data(trace_dir: str) -> None:
    global tdir
    tdir = trace_dir

    ranks = np.arange(0, 512)
    with multiprocessing.Pool(16) as p:
        aggr_df = p.map(read_perf_df, ranks)

    all_dfs = aggr_df
    aggr_df = pd.concat(all_dfs)
    aggr_df.sort_values("usec")
    sum_df = aggr_df.groupby(["rank"], as_index=False).agg(
        {"usec": ["sum", "max", "mean"]}
    )

    sum_df_out = f"{trace_dir}/getslot.csv"
    sum_df.to_csv(sum_df_out, index=None)

    return sum_df


def plot_perf_total(sum_df, trace_dir):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    ranks = np.arange(0, 512)
    ax.plot(ranks, sum_df["usec"]["sum"], label="getslot_sum_us", color="red")

    ax.set_title("getslot(): total_time and max_time")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Avg Time (ms)")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} us".format(x))
    )

    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax.set_ylim(bottom=0)

    fig.legend()
    fig.tight_layout()

    trace_name = os.path.basename(trace_dir)
    plot_fname = f"{trace_name}_perf_getslot_sum"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_perf_avg_max(sum_df, trace_dir):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax2 = ax.twinx()

    ranks = np.arange(0, 512)
    ax.plot(ranks, sum_df["usec"]["mean"], label="getslot_sum_us", color="red")
    ax2.plot(ranks, sum_df["usec"]["max"], label="getslot_max_us", color="blue")

    ax.set_title("getslot(): total_time and max_time")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Avg Time (ms)")
    ax2.set_ylabel("Max Time (ms")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} us".format(x))
    )
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))
    )

    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    fig.legend()
    fig.tight_layout()

    trace_name = os.path.basename(trace_dir)
    plot_fname = f"{trace_name}_perf_getslot_sum_max"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_perf_max(sum_df_a, sum_df_b):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    ranks = np.arange(512)

    ax.plot(ranks, sum_df_a["usec"]["max"], label="getslot_max_us: before")
    ax.plot(ranks, sum_df_b["usec"]["max"], label="getslot_max_us: after")

    ax.set_title("getslot(): max_time before and after")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Max Time (ms")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))
    )

    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax.set_ylim(bottom=0)

    fig.legend()
    fig.tight_layout()

    trace_name = os.path.basename(trace_dir)
    plot_fname = f"{trace_name}_perf_getslot_max_comp"
    PlotSaver.save(fig, "", None, plot_fname)


def run():
    trace_dir = "/mnt/ltio/parthenon-topo/athenapk11"
    sum_df = read_perf_data(trace_dir)
    #  plot_perf_avg_max(sum_df, trace_dir)
    plot_perf_total(sum_df, trace_dir)
    pass


if __name__ == "__main__":
    plot_init()
    run()
