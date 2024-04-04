import glob
import multiprocessing
import numpy as np
import pandas as pd
#  import ipdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import re
import subprocess
import struct
import sys
import time
import os

#  import ray
import traceback

from common import plot_init, PlotSaver, prof_evt_map
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import List, Tuple
from pathlib import Path
#  from task import Task
from trace_reader import TraceReader, TraceOps

from analyze_prof import _fig_make_cax, get_evtmat_by_bid


def plot_heatmap(mat, bounds, norm):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")
    ax.set_title("ABCD")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Timestep")

    fig.tight_layout()

    cax = _fig_make_cax(fig)
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))

    return fig, ax


def plot_compare_prof(
    prof_a: int,
    prof_b: int,
    evt_code: int,
    trace_dir_a: str,
    prof_mat_a: np.ndarray,
    trace_dir_b: str,
    prof_mat_b: np.ndarray,
    clip=0) -> None:
    if clip:
        prof_mat_a = prof_mat_a[:, :clip]
        prof_mat_b = prof_mat_b[:, :clip]

    range_beg = 0
    mat_1d = np.concatenate([prof_mat_a.flatten(), prof_mat_b.flatten()])
    range_end = np.percentile(mat_1d, 99)

    bounds = np.linspace(range_beg, range_end, 10)  # 10 bins
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

    fig, ax = plot_heatmap(prof_mat_a, bounds, norm)
    ax.set_title("Evt{} - Prof{}".format(evt_code, prof_a))
    plot_fname = "heatmap-evt{}-prof{}".format(evt_code, prof_a)
    plot_fname = f"{plot_fname}-clip{clip}"
    PlotSaver.save(fig, trace_dir_a, None, plot_fname)

    fig, ax = plot_heatmap(prof_mat_b, bounds, norm)
    ax.set_title("Evt{} - Prof{}".format(evt_code, prof_b))
    plot_fname = "heatmap-evt{}-prof{}".format(evt_code, prof_b)
    plot_fname = f"{plot_fname}-clip{clip}"
    PlotSaver.save(fig, trace_dir_b, None, plot_fname)

    prof_mat_diff = prof_mat_a - prof_mat_b
    prof_mat_diff_1d = np.abs(prof_mat_diff.flatten())
    diff_max = np.percentile(prof_mat_diff_1d, 99)
    bounds = np.linspace(-diff_max, diff_max, 10)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

    fig, ax = plot_heatmap(prof_mat_diff, bounds, norm)
    ax.set_title("Evt{} - Prof{} minus Prof{}".format(evt_code, prof_a, prof_b))
    plot_fname = "heatmap-evt{}-prof{}minus{}".format(evt_code, prof_a, prof_b)
    plot_fname = f"{plot_fname}-clip{clip}"
    PlotSaver.save(fig, trace_dir_a, None, plot_fname)


def analyze_compare_prof(prof_a, prof_b, evt_code):
    global trace_dir_fmt
    #  prof_a = 22
    #  prof_b = 24
    #  evt_code = 0

    trace_dir_a = trace_dir_fmt.format(prof_a)
    _, prof_mat_a = get_evtmat_by_bid(trace_dir_a, evt_code)

    trace_dir_b = trace_dir_fmt.format(prof_b)
    _, prof_mat_b = get_evtmat_by_bid(trace_dir_b, evt_code)

    plot_compare_prof(
        prof_a,
        prof_b,
        evt_code,
        trace_dir_a,
        prof_mat_a,
        trace_dir_b,
        prof_mat_b,
        clip=0,
    )

    pass

def run_iter():
    for self in [22, 24]:
        for other in [24, 25, 26, 27]:
            if self != other:
                for evtcode in [0, 1]:
                    yield (self, other, evtcode)

def run_parallel():
    with multiprocessing.Pool(8) as pool:
        pool.starmap(analyze_compare_prof, run_iter())


def run():
    analyze_compare_prof(28, 30, 0)
    analyze_compare_prof(28, 30, 1)
    return

    for job in run_iter():
        print(job)
        analyze_compare_prof(*job)


if __name__ == "__main__":
    global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/profile{}"
    run()
    #  run_parallel()
