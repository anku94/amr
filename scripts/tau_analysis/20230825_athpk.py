import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

from matplotlib.ticker import FuncFormatter, MultipleLocator, LogLocator
import multiprocessing
import numpy as np
import pandas as pd
import glob
import re

from common import plot_init_big, PlotSaver, label_map, get_label
from analyze_msgs import aggr_msgs_some, join_msgs

"""
Analyze messages from athenapk5?
"""


def plot_msg_timestamps(trace_name: str):
    chan_df, send_df = aggr_msgs_some(trace_name, 0, 15)
    print(send_df)
    msg_df = join_msgs(chan_df, send_df)
    # print(msg_df)

    dy = send_df["timestamp"].to_numpy()
    dy = dy - min(dy)
    dx = range(len(dy))

    for ts in range(2, 10):
        ts_df = send_df[send_df["ts"] == ts]
        dy = ts_df["timestamp"].to_numpy()
        dy = dy - min(dy)

        hist, bins = np.histogram(dy)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.clear()
        ax.stairs(hist, edges=bins, zorder=2)
        ax.set_title(f"TS: {ts}")
        ax.grid(visible=True, which="major", axis="both", color="#bbb")
        ax.grid(visible=True, which="minor", axis="both", color="#ddd")
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))
        )
        ax.xaxis.set_minor_locator(MultipleLocator(20 * 1e3))
        ax.xaxis.set_major_locator(MultipleLocator(100 * 1e3))
        fig.tight_layout()
        fig.show()
    pass


def get_times_trace(trace_name: str) -> np.ndarray:
    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_name)
    log_fpath = f"{trace_dir}/run/log.txt"
    log_data = open(log_fpath, "r").readlines()
    log_data = [l for l in log_data if "wsec" in l]

    times = []

    for l in log_data:
        mobj = re.search(r"\ wsec_step=([0-9\.e\+\-]+)", l)
        n = float(mobj.group(1))
        times.append(n)

    return np.array(times)


def compare_perf_traces(traces, names):
    times = [ get_times_trace(t) for t in traces ]

    min_len = min([ len(t) for t in times ])
    times = [ t[:min_len][1:] for t in times ]

    dx = np.arange(len(times[0]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for t, name in zip(times, names):
        #  ax.plot(dx, t, label=name)
        ax.plot(dx, t.cumsum(), label=name)

    ymaj, ymin = 0.2, 0.04
    ymaj, ymin = 1000, 100

    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(MultipleLocator(ymin))
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{:.1f} s".format(x)))

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.set_title("Runs - Comparison - 10K")

    #  ax2 = ax.twinx()
    #  ax2.clear()
    #  ty2 = list(zip(times[:-1], times[1:]))
    #  cmap = plt.colormaps["Dark2"]
    #  for idx, (t_old, t_new) in enumerate(ty2):
        #  dy = -(t_new / t_old) + 1
        #  N = 20
        #  dy = np.convolve(dy, np.ones((N,)) / N, mode="valid")
        #  dx = np.arange(len(dy))
        #  label = "improv_pct: {} -> {}".format(idx, idx + 1)
        #  ax2.plot(dx, dy, label=label, color=cmap(idx), linestyle="--")

    #  ax2.yaxis.set_major_formatter(
        #  FuncFormatter(lambda x, pos: "{:.0f}%".format(x * 100))
    #  )
    #  ax2.set_ylim([0, 0.3])
    fig.tight_layout()

    fig.legend(loc="lower center", fontsize=12)

    plot_fname = "ts_comp_" + "_".join(traces)
    PlotSaver.save(fig, "", None, plot_fname)

    pass


def run():
    #  trace_name = "athenapk5"
    #  plot_msg_timestamps(trace_name)

    traces = [
        "athenapk14",
        "athenapk15",
        "athenapk16"
    ]

    names = [ "Baseline", "LPT", "Contig++" ]

    compare_perf_traces(traces, names)

    pass


if __name__ == "__main__":
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    plot_init_big()
    run()
