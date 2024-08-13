import glob
import re

import importlib

parse_preload = importlib.import_module("20240604_parse_preload")
RunSuite = parse_preload.RunSuite
get_runs = parse_preload.get_runs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import plot_init_big, PlotSaver
from typing import Literal

Policy = Literal["baseline", "cdp", "hybrid25", "hybrid50", "hybrid75", "lpt"]

PolicyFuncData = np.ndarray
PolicyData = dict[str, PolicyFuncData]
AllPolicyData = dict[Policy, PolicyData]


def read_amfile(amfile: str) -> PolicyData:
    with open(amfile, "r") as f:
        lines = f.readlines()

    headings = lines[0::2]
    data = lines[1::2]

    headings = [h.strip() for h in headings]
    data = [np.array(d.strip().split(","), dtype=float) for d in data]

    assert len(headings) == len(data)
    amrmon_data = {h: d for h, d in zip(headings, data)}

    return amrmon_data


def read_all_amfiles(trace_prefix: str, policies: list[Policy]) -> AllPolicyData:
    trace_dir_fmt = "/mnt/ltio/parthenon-topo"

    amrmon_files = [
        f"{trace_dir_fmt}/{trace_prefix}.{policy}/trace/amrmon_rankwise.txt"
        for policy in policies
    ]

    amrmon_data: AllPolicyData = {
        policy: read_amfile(amfile) for policy, amfile in zip(policies, amrmon_files)
    }

    return amrmon_data


def read_run_suite(suite: RunSuite) -> AllPolicyData:
    policies = suite.trace_names
    amrmon_files = suite.amrmon_files

    amrmon_data: AllPolicyData = {
        policy: read_amfile(amfile) for policy, amfile in zip(policies, amrmon_files)
    }

    return amrmon_data


def get_data_for_key(
    amrmon_data: AllPolicyData, policies: list[Policy], key: str
) -> list[np.ndarray]:
    data = [amrmon_data[policy][key] for policy in policies]

    return data


def find_matching_func(amrmon_data: AllPolicyData, func: str) -> str | None:
    for _, data in amrmon_data.items():
        for key in data.keys():
            if func in key:
                return key

    return None


def plot_func_data(
    data: list[np.ndarray], policies: list[Policy], trace_prefix: str, func: str
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for d, policy in zip(data, policies):
        ax.plot(d / 1e6, label=policy, zorder=2)

    ax.set_title(f"Rankwise times for {func}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (s)")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} s"))
    ax.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")

    ax.legend()
    fig.tight_layout()

    func_sanitized = func.lower()
    func_sanitized = re.sub(r"[^a-z0-9]", "", func_sanitized)

    plot_fname = f"pprof.rankwise.{trace_prefix}.{func_sanitized}"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_funcs_stacked(
    nranks: int, run_name: str, funcs: list[str], data: list[np.ndarray[float]]
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    data = np.stack(data, axis=0) / 1e6

    dx = np.arange(nranks)
    ax.stackplot(dx, data, labels=funcs, zorder=2)
    ax.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.set_title(f"Stacked rankwise times for {run_name}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} s"))

    ax.legend(ncol=10, fontsize=6)
    fig.tight_layout()

    ax.set_ylim([0, 10000])

    plot_fname = "pprof.rankwise.stacked." + run_name
    PlotSaver.save(fig, "", None, plot_fname)


def run_stacked():
    nranks = 4096
    run_id = f"blastw{nranks}.04"
    fallback_run_id = f"blastw{nranks}.03"

    suite = get_runs(nranks, run_id, fallback_run_id)
    print(f"\n{suite}")

    all_data = read_run_suite(suite)
    run_names = suite.trace_names

    funcs = [
        ("Task_FillDerived", "FD"),
        ("ConToPrim::Solve", "C2P"),
        # ("LoadBalancingAndAdaptiveMeshRefinement", "LB&AMR"),
        ("UpdateMeshBlockTree", "UMBT"),
        ("RedistributeAndRefineMeshBlocks", "RRMB"),
        ("Task_LoadAndSendBoundBufs", "BBSend"),
        ("Task_ReceiveBoundBufs", "BBRecv"),
        ("MPI_Allreduce", "Allreduce"),
    ]

    func_names = [f[0] for f in funcs]
    func_labels = [f[1] for f in funcs]

    for cur_run in run_names:
        data_run = all_data[cur_run]
        data_funcs = [data_run[func] for func in func_names]
        np.stack(data_funcs, axis=0).shape
        plot_name = re.sub(r"[^a-z0-9]", "", cur_run.lower())
        plot_funcs_stacked(nranks, plot_name, func_labels, data_funcs)


def run():
    nranks = 4096
    run_id = f"blastw{nranks}.04"
    fallback_run_id = f"blastw{nranks}.03"

    suite = get_runs(nranks, run_id, fallback_run_id)
    print(f"\n{suite}")

    all_data = read_run_suite(suite)
    policies = suite.trace_names
    trace_prefix = suite.run_id

    funcs = [
        # "Driver_Main",
        # "RedistributeAndRefineMeshBlocks",
        # "CalculateLoadBalance",
        # "Mesh::Initialize",
        "LoadBalancingAndAdaptiveMeshRefinement",
        "MPI_Allgather",
        "MPI_Allreduce",
        "Task_FillDerived"
    ]

    funcs_stacked = [
        "Task_FilDerived",
        "LoadBalancingAndAdaptiveMeshRefinement",
    ]

    for func in funcs:
        data = get_data_for_key(all_data, policies, func)
        plot_func_data(data, policies, trace_prefix, func)

    return

    func_patts = [f"Step {x}" for x in range(2, 10)]

    for patt in func_patts:
        func = find_matching_func(all_data, patt)

        if func is None:
            continue

        print(f"Plotting: {func}")

        func_data = get_data_for_key(all_data, policies, func)
        plot_func_data(func_data, policies, trace_prefix, func)


if __name__ == "__main__":
    run()
    # run_stacked()
