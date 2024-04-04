import glob
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import re
import subprocess
import struct
import sys
import time
import os

from common import plot_init, PlotSaver, prof_evt_map
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import List, Tuple
from pathlib import Path
from trace_reader import TraceReader, TraceOps


def get_prof_path(trace_dir: str, evt: int) -> str:
    # replace merged with agg if single self-contained run
    ppath = f"{trace_dir}/prof.merged.evt{evt}.csv"
    ppath = f"{trace_dir}/prof.aggr.evt{evt}.csv"
    return ppath


def make_uniform(obj_2d):
    lens = list(map(lambda x: x.shape[0], obj_2d))
    max_len = max(lens)
    padded_1d = [np.pad(arr, (0, max_len - len(arr)), "constant") for arr in obj_2d]
    mat = np.stack(padded_1d)

    print(f"Mat shape: {mat.shape}")

    return mat


def get_evt_mat(evt):
    clip = False

    global trace_dir
    df_path = get_prof_path(trace_dir, evt)
    print(f"Reading dataframe: {df_path}")
    df = pd.read_csv(df_path)

    df_agg = df.groupby(["sub_ts", "block_id"], as_index=False).agg(
        {"rank": "min", "time_us": ["sum", "count"]}
    )

    df_agg.columns = list(map(lambda x: "_".join(x).strip("_"), df_agg.columns))

    df_agg2 = df_agg.groupby("sub_ts", as_index=False).agg(
        {"time_us_sum": list, "time_us_count": list}
    )

    if clip:
        match_low = df_agg2["sub_ts"] >= 1155
        match_hi = df_agg2["sub_ts"] <= 6270
        df_clip = df_agg2[match_low & match_hi].copy()
    else:
        df_clip = df_agg2

    mat_sum = df_clip["time_us_sum"].apply(np.array).to_numpy()
    mat_sum = make_uniform(mat_sum)

    mat_count = df_clip["time_us_count"].apply(np.array).to_numpy()
    mat_count = make_uniform(mat_count)

    return mat_sum, mat_count


def get_regr_slopes(mat):
    mat = mat.T
    m, n = mat.shape
    X = np.vstack([np.arange(n)] * m)

    # slope = N sum(xy) - sum(x)sum(y) / nsum(x^2) - sum(x)^2
    den = n * (X**2).sum(axis=1) - X.sum(axis=1) ** 2
    num1 = n * np.sum(np.multiply(mat, X), axis=1)
    num2 = np.multiply(X.sum(axis=1), mat.sum(axis=1))

    slopes = (num1 - num2) / den
    indices = np.argsort(slopes)[::-1]
    slopes_sorted = slopes[indices]
    slopes_tuple = list(zip(slopes_sorted, indices))
    return slopes_tuple


def plot_blocks(ax, mat, block_idxes, key):
    global trace_dir

    data_x = np.arange(mat.shape[0]) + 1155

    for idx in block_idxes:
        data_y = mat[:, idx]
        ax.plot(data_x, data_y, "o", label=f"b{idx}", ms=1)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"({key})")

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))
    ax.set_ylim([0, 40000])
    ax.yaxis.set_major_locator(MultipleLocator(6000))
    ax.yaxis.set_minor_locator(MultipleLocator(1500))
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")

    #  ax.legend(ncol=4)


def plot_blocks_evt(mat, evt):
    slopes = get_regr_slopes(mat)

    idxes = list(map(lambda x: x[1], slopes))
    idxes_top8 = idxes[:8]
    idxes_top100 = idxes[:100:13]
    idxes_all = idxes[::121]

    fig = plt.figure(figsize=(9, 5))
    axes = fig.subplots(1, 3)

    plot_blocks(axes[0], mat, idxes_top8, "Top 8")
    plot_blocks(axes[1], mat, idxes_top100, "Top 100")
    plot_blocks(axes[2], mat, idxes_all, "All")

    fig.suptitle(f"Selected Blocks vs Selected Timeslice (Evt {evt})", fontsize=18)
    fig.tight_layout()

    plot_fname = f"interesting_blocks.evt{evt}"
    PlotSaver.save(fig, trace_dir, None, plot_fname)


def plot_blocks_imshow(mat, evt, evt_label, vmin, vmax):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    #  norm = colors.LogNorm(vmin=15000, vmax=30000)
    #  norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    bounds = np.linspace(vmin, vmax, 256)
    bounds = np.linspace(vmin, vmax, 64)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="max")

    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")
    #  im = ax.imshow(mat, aspect="auto", cmap="plasma")
    #  evts = {0: "FillDerived", 1: "CalculateFluxes"}
    #  evt_label = evts[evt]
    ax.set_title(f"Block ID vs Time (Evt {evt_label})")
    ax.set_xlabel("Block ID")
    ax.set_ylabel("Timestep")

    #  ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x + 1155)}")

    fig.tight_layout()

    fig.subplots_adjust(left=0.15, right=0.78)
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    #  cax.yaxis.set_major_formatter(FuncFormatter(cax_fmt))
    #  cax.xaxis.set_major_formatter(FuncFormatter(cax_fmt))

    #  cbar_ticks = np.arange(15, 34, 5) * 1e3
    #  cbar_labels = list(map(lambda x: cax_fmt(x, None), cbar_ticks))
    cbar = fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))
    #  cbar = fig.colorbar(im, cax=cax, format="{x:.0f} ms")
    #  cbar = fig.colorbar(im, format=FuncFormatter(cax_fmt))
    #  cbar.ax.ticklabel_format(style='plain')
    #  cbar.ax.yaxis.set_major_formatter(FuncFormatter(cax_fmt))
    #  cbar.ax.xaxis.set_major_formatter(FuncFormatter(cax_fmt))
    #  cax.yaxis.set_major_formatter(FuncFormatter(cax_fmt))
    #  cax.xaxis.set_major_formatter(FuncFormatter(cax_fmt))
    #  cbar = fig.colorbar(im, cax=cax)
    #  cbar.ax.set_yticks(cbar_ticks)
    #  cbar.ax.set_yticklabels(cbar_labels)

    global trace_dir
    fname = f"blockmat.sums.evt{evt}"
    PlotSaver.save(fig, trace_dir, None, fname)


def plot_blocks_imshow_v2(mat, evt, evt_label, vmin, vmax):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    #  bounds = np.linspace(vmin, vmax, 256)
    bounds = np.linspace(vmin, vmax, 64)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="max")

    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")
    #  ax.set_title(f"Block ID vs Time (Evt {evt_label})")
    ax.set_xlabel("Timestep", fontsize=15)
    ax.set_ylabel("Block ID", fontsize=15)

    ax.tick_params(axis="both", labelsize=13)

    tick_max = mat.shape[0]
    ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x/1e3)}K")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{int(tick_max - x)}")

    fig.tight_layout()

    left_bound = 0.15
    right_bound = 0.72
    fig.subplots_adjust(left=left_bound, right=right_bound)
    cax = fig.add_axes([right_bound + 0.03, 0.17, 0.07, 0.76])
    cax.tick_params(axis="both", labelsize=13)
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    cbar = fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))
    cax.set_ylabel("Kernel Invocation Time (ms)", fontsize=16)
    global trace_dir
    fname = f"blockmat.sums.evt{evt}.v2"
    PlotSaver.save(fig, trace_dir, None, fname)
    pass


def plot_counts_imshow(mat, evt):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    norm = colors.LogNorm(vmin=15000, vmax=30000)
    bounds = np.arange(0, 6)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="max")

    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")
    evts = {0: "FillDerived", 1: "CalculateFluxes"}
    evt_label = evts[evt]
    ax.set_title(f"Block ID vs InvokeCount-Per-TS (Evt {evt_label})")
    ax.set_xlabel("Block ID")
    ax.set_ylabel("Timestep")

    ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x + 1155)}")

    fig.tight_layout()

    fig.subplots_adjust(left=0.15, right=0.78)
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])

    #  cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    #  cbar_ticks = np.arange(15, 34, 5) * 1e3
    #  cbar_labels = list(map(lambda x: cax_fmt(x, None), cbar_ticks))
    #  cbar = fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))
    cbar = fig.colorbar(im, cax=cax)
    #  cbar.ax.set_yticks(cbar_ticks)
    #  cbar.ax.set_yticklabels(cbar_labels)

    global trace_dir
    fname = f"blockmat.counts.evt{evt}"
    PlotSaver.save(fig, trace_dir, None, fname)


def plot_mat_experiments():
    mat0_sum, mat0_count = get_evt_mat(0)
    ms = mat0_sum.copy()
    ms = ms.astype(float)
    ms[ms == 0] = np.nan

    vmin = np.nanpercentile(ms, 2)
    vmax = np.nanpercentile(ms, 98)

    plot_blocks_imshow(mat0_sum, 0, "FillDerived", 15000, 25000)

    plot_blocks_imshow(ms, 0, "FillDerived", vmin, vmax)
    ms
    ms.T.shape
    rot_ms = ms.T[::-1]
    plot_blocks_imshow_v2(rot_ms, 0, "FillDerived", vmin, vmax)
    pass


def get_interesting_blocks():
    mat0_sum, mat0_count = get_evt_mat(0)
    #  plot_blocks_evt(mat0_sum, 0)
    plot_blocks_imshow(mat0_sum, 0)

    plot_counts_imshow(mat0_count, 0)

    mat1_sum, mat1_count = get_evt_mat(1)
    #  plot_blocks_evt(mat1_sum, 1)
    plot_blocks_imshow(mat1_sum, 1)
    plot_counts_imshow(mat1_count, 1)


def get_interesting_blocks_2():
    evts = [0, 1, 5, 6]
    evt_labels = ["FD", "CF", "SNF", "TabCool"]

    evts = evts[:2]
    evt_labels = evt_labels[:2]

    evt_tuples = list(map(get_evt_mat, evts))
    mats = list(map(lambda x: x[0], evt_tuples))
    mat_sum = np.sum(mats, axis=0)

    for mat, evt, label in zip(mats, evts, evt_labels):
        vmin = np.percentile(mat, 2)
        vmax = np.percentile(mat, 98)
        print(f"Evt: {evt}, Vmin: {vmin}, Vmax: {vmax}")
        plot_blocks_imshow(mat, evt, label, vmin, vmax)

    vmin = np.percentile(mat_sum, 1)
    vmax = np.percentile(mat_sum, 99)
    plot_blocks_imshow(mat_sum, "0156", "COMP_ALL", vmin, vmax)
    pass


def run():
    plot_init()
    get_interesting_blocks_2()


if __name__ == "__main__":
    global trace_dir_fmt
    global trace_dir

    trace_dir_fmt = "/mnt/ltio/parthenon-topo/profile{}"
    trace_dir = "/mnt/ltio/parthenon-topo/profile40"
    trace_dir = "/mnt/ltio/parthenon-topo/profile37"
    trace_dir = "/mnt/ltio/parthenon-topo/profile39"
    trace_dir = "/mnt/ltio/parthenon-topo/burgers2"
    trace_dir = "/mnt/ltio/parthenon-topo/athenapk1"
    trace_dir = "/mnt/ltio/parthenon-topo/stochsg2"
    run()
