import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import glob
import os
import re

from common import plot_init, PlotSaver, label_map, get_label, profile_label_map
from trace_reader import TraceReader, TraceOps
from typing import Dict, List


def parse_array(arr):
    arr = arr.strip("[]").split(",")
    arr = [int(i) for i in arr]
    return arr

def read_df(trace_dir, evt_code):
    df_path = "{trace_dir}/prof.aggr.evt{evt_code}.csv"
    df = pd.read_csv(df_path)
    pass


def get_nblocks(trace_dir: str):
    df_path = f"{trace_dir}/prof.aggrmore.evt3.csv"
    df = pd.read_csv(df_path)
    df["bl_cnt"] = df["block_id"].apply(lambda x: len(parse_array(x)))
    nblocks = df["bl_cnt"].to_numpy()
    #  aggr_df = df.groupby("sub_ts", as_index=False).agg({"block_id": "count"})
    #  nblocks = aggr_df["block_id"].to_numpy()
    return nblocks


def plot_nblocks(trace_dirs: List[str]):
    # set a primary trace dir, assumed to be idx 0
    # used by PlotSaver to name things
    global trace_dir
    trace_dir = trace_dirs[0]

    all_nblocks = [get_nblocks(t) for t in trace_dirs]
    data_x = np.arange(len(all_nblocks[0]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    for tidx, tdir in enumerate(trace_dirs):
        data_y = all_nblocks[tidx]
        label = profile_label_map[os.path.basename(tdir)]

        if tidx == 0:
            linewidth = 2
            zorder = 3
            alpha = 1
        else:
            linewidth = 1
            zorder = 2
            alpha = 0.6

        ax.plot(
            data_x, data_y, linewidth=linewidth, label=label, alpha=alpha, zorder=zorder
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Block Count")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f}K".format(x / 1e3))

    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_minor_locator(MultipleLocator(125))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    fig.tight_layout()

    ax.legend()

    plot_fname = "nblocks"
    PlotSaver.save(fig, "", None, plot_fname)


def run_plot():
    trace_dir_pref = "/mnt/ltio/parthenon-topo/profile"
    trace_nums = [23, 27]
    trace_nums = [37, 38, 39]
    trace_dirs = [f"{trace_dir_pref}{tnum}" for tnum in trace_nums]
    plot_nblocks(trace_dirs)
    pass


if __name__ == "__main__":
    plot_init()
    run_plot()
