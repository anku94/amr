import importlib
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from typing import TypedDict
from common import PlotSaver

_tmp = importlib.import_module("20240704_analyze_runlogs")
ParsedLog = _tmp.ParsedLog
read_log = _tmp.read_log
read_log_diff_from_prev = _tmp.read_log_diff_from_prev

trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"


def get_trace_file(trace_name: str) -> str:
    global trace_dir_fmt

    cycles_log = f"{trace_name}/run/cycles.log"
    run_log = f"{trace_name}/run/log.txt"

    cycles_fpath = trace_dir_fmt.format(cycles_log)
    run_fpath = trace_dir_fmt.format(run_log)

    print(f"Checking for {cycles_fpath} or {run_fpath}")

    if os.path.exists(cycles_fpath):
        return cycles_fpath

    if os.path.exists(run_fpath):
        return run_fpath

    raise FileNotFoundError(f"Could not find trace file for {trace_name}")


def get_trace_label(trace_fpath: str) -> str:
    trace_chunks = trace_fpath.split("/")
    assert trace_chunks[-2] == "run"
    trace_name = trace_chunks[-3]
    trace_policy = trace_name.split(".")[-1]
    return trace_policy


def plot_comp(trace_fpaths: list[tuple[str, str]]) -> None:
    # make fig with constrained layout
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")

    for idx, (fpath1, fpath2) in enumerate(trace_fpaths):
        log1 = read_log_diff_from_prev(fpath1)
        log2 = read_log_diff_from_prev(fpath2)

        allts1 = np.array(log1["time_allts"]).cumsum()
        allts2 = np.array(log2["time_allts"]).cumsum()
        min_len = min(len(allts1), len(allts2))
        allts1 = allts1[:min_len]
        allts2 = allts2[:min_len]

        allts1[-1]
        allts2[-1]

        color = f"C{idx}"

        label1 = get_trace_label(fpath1)
        label2 = get_trace_label(fpath2)

        if label2 == label1:
            label2 = ""

        ax.plot(allts1, label=label1, color=color, linestyle="--")
        ax.plot(allts2, label=label2, color=color, linestyle="-")

    ax.set_title("BlastWave4096 - CDP vs CDPC512 - Comparison")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which="major", axis="both", color="#bbb")
    ax.grid(which="minor", axis="y", color="#ddd")

    ax.legend(ncol=2)

    ax.set_axisbelow(True)

    plot_fname = "blastw03vs04cdpc512"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_comp_2x2(trace_fpaths: list[tuple[str, str]]) -> None:
    # make fig with constrained layout
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    axls = axes.flatten()

    for idx, (fpath1, fpath2) in enumerate(trace_fpaths):
        ax = axls[idx]

        log1 = read_log_diff_from_prev(fpath1)
        log2 = read_log_diff_from_prev(fpath2)

        allts1 = np.array(log1["time_allts"]).cumsum()
        allts2 = np.array(log2["time_allts"]).cumsum()
        min_len = min(len(allts1), len(allts2))
        allts1 = allts1[:min_len]
        allts2 = allts2[:min_len]

        allts1[-1]
        allts2[-1]

        color = f"C{idx}"

        label1 = get_trace_label(fpath1)
        label2 = get_trace_label(fpath2)

        if label2 == label1:
            label2 = ""

        # ax.plot(allts1, label=label1, color=color, linestyle="--")
        # ax.plot(allts2, label=label2, color=color, linestyle="-")
        ax.plot(allts1 - allts2, label=f"{label1} - {label2}", color=color)

    fig.suptitle("BlastWave4096 - CDP vs CDPC512 - Comparison")
    fig.supxlabel("Timestep")
    fig.supylabel("Time (s)")

    all_ylims = list(zip(*[ax.get_ylim() for ax in axls]))
    max_ylim = (min(all_ylims[0]), max(all_ylims[1]))
    all_xlims = list(zip(*[ax.get_xlim() for ax in axls]))
    max_xlim = (min(all_xlims[0]), max(all_xlims[1]))

    for ax in axls:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} s"))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.grid(which="major", axis="both", color="#bbb")
        ax.grid(which="minor", axis="y", color="#ddd")

        ax.legend(ncol=2)

        ax.set_axisbelow(True)
        ax.set_xlim(max_xlim)
        ax.set_ylim(max_ylim)

    axes[0][0].tick_params(labelbottom=False)
    axes[0][1].tick_params(labelleft=False, labelbottom=False)
    axes[1][1].tick_params(labelleft=False)

    plot_fname = "blastw03vs04cdpc512.2x2.diff"
    PlotSaver.save(fig, "", None, plot_fname)


def run_comp():
    traces_to_comp = [
        ("blastw4096.03.cdp", "blastw4096.04.cdpc512"),
        ("blastw4096.03.hybrid25", "blastw4096.04.hybrid25"),
        ("blastw4096.03.hybrid50", "blastw4096.04.hybrid50"),
        ("blastw4096.03.hybrid75", "blastw4096.04.hybrid75"),
    ]
    trace_paths = [
        (get_trace_file(t1), get_trace_file(t2)) for t1, t2 in traces_to_comp
    ]

    # plot_comp(trace_paths)
    plot_comp_2x2(trace_paths)


def run():
    run_comp()


if __name__ == "__main__":
    run()
