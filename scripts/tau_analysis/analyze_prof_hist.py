import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from common import plot_init, PlotSaver, prof_evt_map
from matplotlib.ticker import MultipleLocator

def plot_hist(trace_idx, evt_id):
    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_idx)
    evt_df_path = f"{trace_dir}/prof.aggr.evt{evt_id}.csv"
    evt_df = pd.read_csv(evt_df_path)

    bins = np.arange(0, 25, 0.25)
    data_hist = evt_df["time_us"] / 1e3
    hist, bin_edges = np.histogram(data_hist, bins=bins)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.hist(data_hist, bins, zorder=2)
    ax.set_title(f"Evt Time Distrib - Trace {trace_idx}, Evt {evt_id}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Counts")

    ax.yaxis.set_major_locator(MultipleLocator(10 * 1e6))
    ax.yaxis.set_minor_locator(MultipleLocator(2 * 2e6))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f}M".format(x/1e6))

    fig.tight_layout()
    plot_fname = f"hist.evt{evt_id}.profile{trace_idx}"
    PlotSaver.save(fig, "", None, plot_fname)

def run():
    plot_hist(31, 0)
    plot_hist(31, 1)

if __name__ == "__main__":
    global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/profile{}"
    run()
