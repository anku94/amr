import os
import glob
import multiprocessing
import numpy as np
import pandas as pd
import io
import re
from common import plot_init, plot_init_big, PlotSaver
from typing import get_args, Callable, Literal, Tuple, TypedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path

#  from task import Task
from trace_reader import TraceOps
from trace_common import TraceUtils, SingleTrace

global trace_dir_fmt
trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
trace_dir_fmt = "/proj/TableFS/ankushj/amr-jobs/{}"

"""
Returns all unique leaf events in the set
Discards intermediate events if leaf events present
"""

PPROF_CACHE = {}

ProfPhase = str
SingleSpec = Tuple[str, Callable[[str], bool], ProfPhase]


class StackedPlotSpec(TypedDict):
    trace_key: str
    nranks: int
    # keys that begin with _ will be plotted as lines
    keys: list[str]
    data: dict[str, np.ndarray]
    ylim_top: int
    plot_fname: str


class BarPlotSpec(TypedDict):
    trace_key: str
    trace_names: list[str]
    nranks: int
    keys: list[str]
    data: dict[str, list]
    plot_fname: str


def send_plot(fig):
    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Encode the image to base64
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plot_data

    # Send the plot data to the server
    response = requests.post(
        "http://127.0.0.1:5000/update_plot",
        json={"plot_data": plot_data},
        proxies={"http": None, "https": None},
    )

    if response.status_code == 200:
        print("Plot updated successfully")
    else:
        print("Failed to update plot")


def get_cache_key(trace_key: str, trace: SingleTrace) -> str:
    return f"{trace_key}_{trace.name}"


def read_pprof(fpath: str):
    f = open(fpath).readlines()
    lines = [l.strip("\n") for l in f if l[0] != "#"]

    nfuncs = int(re.findall(r"(\d+)", lines[0])[0])
    rel_lines = lines[1:nfuncs]
    prof_cols = [
        "name",
        "ncalls",
        "nsubr",
        "excl_usec",
        "incl_usec",
        "unknown",
        "group",
    ]
    df = pd.read_csv(io.StringIO("\n".join(rel_lines)), sep="\s+", names=prof_cols)

    rank = re.findall(r"profile\.(\d+)\.0.0$", fpath)[0]
    df["rank"] = int(rank)
    return df


def read_all_pprof(trace: SingleTrace):
    prof_dir = trace.get_tau_dir()
    pprof_glob = f"{prof_dir}/profile.*"
    all_files = glob.glob(pprof_glob)

    print(f"-----\nTrace dir: {trace.name}, reading {len(all_files)} files")

    all_files[0]
    read_pprof(all_files[1])

    with multiprocessing.Pool(16) as pool:
        all_dfs = pool.map(read_pprof, all_files)

    concat_df = pd.concat(all_dfs)
    #  del all_dfs
    concat_df.sort_values(["rank"], inplace=True)
    concat_df["name"] = concat_df["name"].str.strip()
    return concat_df


def read_all_pprof_cached(trace_key: str, trace: SingleTrace):
    cache_key = get_cache_key(trace_key, trace)
    if cache_key in PPROF_CACHE:
        return PPROF_CACHE[cache_key]
    else:
        concat_df = read_all_pprof(trace)
        PPROF_CACHE[cache_key] = concat_df
        return concat_df


def filter_pprof_by_func(
    concat_df: pd.DataFrame, func: Callable[[str], bool], nranks: int
) -> np.ndarray:
    mask = concat_df["name"].apply(func)
    df_func = concat_df[mask].sort_values("rank")
    matching_names = df_func["name"].unique()

    print(f"Found {len(matching_names)} matching functions")
    print(" - " + "\n - ".join(matching_names[:]) + "\n...")

    df_aggr = df_func.groupby("rank").agg({"incl_usec": "sum"})
    df_aggr = df_aggr.reindex(range(nranks), fill_value=0)
    missing_ranks = df_aggr[df_aggr["incl_usec"] == 0]
    max_rank = df_aggr.index.max()

    if len(missing_ranks) > 0:
        print(f"WARNING: {len(missing_ranks)} ranks missing, max rank: {max_rank}")

    return df_aggr["incl_usec"].to_numpy()


def gather_pprof_from_spec(
    concat_df: pd.DataFrame, stacked_spec: list, nranks: int
) -> dict[str, np.ndarray]:
    pprof_data = {}

    for spec in stacked_spec:
        name, func, _ = spec
        data = filter_pprof_by_func(concat_df, func, nranks)
        if sum(data) == 0:
            print(f"INFO: No data for {name}, skipping ...")
            continue

        pprof_data[name] = data

    return pprof_data


def gather_pprof_from_spec_cached(
    trace_key: str, trace: SingleTrace, stacked_spec: list[SingleSpec], nranks: int
) -> dict[str, np.ndarray]:
    pprof_data = {}

    spec_misses: list[SingleSpec] = []
    spec_ckeys: list[str] = []

    for s in stacked_spec:
        singletrace_key = get_cache_key(trace_key, trace)
        cache_key = f"{singletrace_key}.{s[0]}"
        if cache_key in PPROF_CACHE:
            pprof_data[s[0]] = PPROF_CACHE[cache_key]
        else:
            spec_misses.append(s)
            spec_ckeys.append(cache_key)

    concat_df = read_all_pprof_cached(trace_key, trace)
    missed_data = gather_pprof_from_spec(concat_df, spec_misses, nranks)

    for k, ck in zip(missed_data.keys(), spec_ckeys):
        PPROF_CACHE[ck] = missed_data[k]
        pprof_data[k] = missed_data[k]

    return pprof_data


def plot_stacked_plot(spspec: StackedPlotSpec) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    valid_keys = [k for k in spspec["keys"] if k in spspec["data"]]
    keys_stackplot = [k for k in valid_keys if not k.startswith("_")]
    keys_lineplot = [k.strip("_") for k in valid_keys if k.startswith("_")]

    data_stackplot = [spspec["data"][k] for k in keys_stackplot]
    ax.stackplot(
        range(spspec["nranks"]), *data_stackplot, labels=keys_stackplot, zorder=2
    )

    for k in keys_lineplot:
        kx = [0, spspec["nranks"]]
        kymax = np.max(spspec["data"][f"_{k}"])
        ky = [kymax, kymax]
        ax.plot(kx, ky, label=k, linestyle="--", linewidth=2, zorder=2)

    if spspec["ylim_top"] > 0:
        ax.set_ylim(0, spspec["ylim_top"] * 1e6)
    else:
        ax.set_ylim(bottom=0)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")

    ax.legend(loc="upper left", ncol=8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (s)")

    fig.tight_layout()

    send_plot(fig)
    # PlotSaver.save(fig, "", None, spspec["plot_fname"])


def gen_bar_plot_spec(
    trace_key: str, phases: list[str], spec: list[SingleSpec]
) -> BarPlotSpec:
    suite = TraceUtils.get_traces(trace_key)
    ntraces = len(suite.traces)

    bpspec: BarPlotSpec = {
        "trace_key": trace_key,
        "trace_names": suite.trace_names(),
        "nranks": suite.nranks,
        "keys": phases,
        "data": {p: [0] * ntraces for p in phases},
        "plot_fname": f"pprof.bar",
    }

    all_data_summed = []

    for trace in suite.traces:
        stacked_data = gather_pprof_from_spec_cached(
            trace_key, trace, spec, suite.nranks
        )
        stacked_data_summed = {k: np.sum(v) for k, v in stacked_data.items()}
        all_data_summed.append(stacked_data_summed)

    spec_phase_map: dict[str, ProfPhase] = {s: p for s, _, p in spec}

    for tidx, tdata in enumerate(all_data_summed):
        for func, val in tdata.items():
            val_avg = val / suite.nranks
            phase = spec_phase_map[func]
            tfdata = bpspec["data"]

            if phase not in phases:
                continue

            tfdata[phase][tidx] += val_avg

    return bpspec


def plot_bar_plot_spec(bpspec: BarPlotSpec) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    trace_key = bpspec["trace_key"]

    keys = bpspec["keys"]
    nkeys = len(keys)

    traces = bpspec["trace_names"]
    ntraces = len(traces)
    bar_width = 1 / (ntraces + 1)

    data = [bpspec["data"][k] for k in keys]
    data_t = list(zip(*data))

    dx = np.arange(nkeys)

    for tidx, trace in enumerate(traces):
        dy = data_t[tidx]
        dy_labels = [f"{v/1e6:.0f}" if v > 0 else "" for v in dy]
        bars = ax.bar(
            dx + tidx * bar_width,
            dy,
            width=bar_width,
            label=trace,
            align="center",
            zorder=2,
        )
        ax.bar_label(bars, labels=dy_labels, padding=3)

    ax.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f} s"))

    ax.set_ylabel("Time (s)")
    ax.set_xticks(dx + 0.5)
    ax.set_xticklabels(keys)
    ax.set_title(f"Phase Times for {trace_key}")

    ax.legend()

    plot_fname_pref = bpspec["plot_fname"]
    plot_fname = f"{plot_fname_pref}.{trace_key}"
    send_plot(fig)
    # PlotSaver.save(fig, "", None, plot_fname)


def plot_spec_stacked_inner(
    trace: SingleTrace,
    pprof_spec: list[SingleSpec],
    plot_spec: StackedPlotSpec,
) -> None:
    trace_key = plot_spec["trace_key"]
    nranks = plot_spec["nranks"]
    plot_fname = plot_spec["plot_fname"]

    stacked_data = gather_pprof_from_spec_cached(trace_key, trace, pprof_spec, nranks)

    spspec: StackedPlotSpec = {
        "trace_key": trace_key,
        "nranks": nranks,
        "keys": plot_spec["keys"],
        "data": stacked_data,
        "ylim_top": plot_spec["ylim_top"],
        "plot_fname": f"{plot_fname}.{trace.name}",
    }

    plot_stacked_plot(spspec)


def gen_spec_lb_steps() -> list[SingleSpec]:
    pprof_spec: list[SingleSpec] = [
        ("S1", lambda x: x.startswith("Step 1"), "Any"),
        ("S2", lambda x: x.startswith("Step 2"), "Any"),
        ("S3", lambda x: x.startswith("Step 3"), "Any"),
        ("S4", lambda x: x.startswith("Step 4"), "Any"),
        ("S5", lambda x: x.startswith("Step 5"), "Any"),
        ("S6", lambda x: x.startswith("Step 6"), "Any"),
        ("S7", lambda x: x.startswith("Step 7"), "Any"),
        ("S8", lambda x: x.startswith("Step 8"), "Any"),
        ("S9", lambda x: x.startswith("Step 9"), "Any"),
    ]

    return pprof_spec


""" Instrumented variant of Parthenon, with barriers added to LB&AMR """


def gen_spec_lb_instr() -> list[SingleSpec]:
    mpibarfunc = "taupreload_main => Driver_Main => LoadBalancingAndAdaptiveMeshRefinement => RedistributeAndRefineMeshBlocks => MPI_Barrier()"

    spec_args = [
        ["_rnr", "RedistributeAndRefineMeshBlocks"],
        ["s5", "Step 5: Allocate send and recv buf"],
        ["s6", "Step 6: Pack buffer and start recv"],
        ["s7", "Step 7: Pack and send buffers"],
        ["s8", "Step 8: Construct new MeshBlockList"],
        ["s9", "Step 9: Recv data and unpack"],
        ["clb", "CalculateLoadBalance"],
        ["init", "Mesh::Initialize"],
        ["mpibr", mpibarfunc],
    ]

    def gen(a: str, b: str) -> SingleSpec:
        return (a, lambda x: x.strip() == b, a)

    spec: list[SingleSpec] = [gen(*arg) for arg in spec_args]
    return spec


def gen_spec_all_mpi() -> list[SingleSpec]:
    def ss(a: str) -> SingleSpec:
        return (a, lambda x: x.strip() == a, a)

    spec_args = [
        "MPI Collective Sync",
        "MPI_Allgather()",
        "MPI_Barrier()",
        "MPI_Init()",
        "MPI_Test()",
        "MPI_Wait()",
        "MPI_Comm_dup()",
        "MPI_Iprobe()",
        "MPI_Isend()",
    ]

    spec: list[SingleSpec] = [ss(arg) for arg in spec_args]

    return spec


def gen_spec_std() -> list[SingleSpec]:
    spec: list[SingleSpec] = [
        ("_Total", lambda x: x.strip() == "Driver_Main", "App"),
        ("C2P::S", lambda x: x.strip() == "ConToPrim::Solve", "Comp"),
        ("CalcFlx", lambda x: x.strip() == "CalculateFluxes", "Comp"),
        ("WSD", lambda x: x.strip() == "WeightedSumData", "Comp"),
        ("SendBB", lambda x: x.strip() == "Task_LoadAndSendBoundBufs", "P2P"),
        ("RecvBB", lambda x: x.strip() == "Task_ReceiveBoundBufs", "P2P"),
        ("R&RMB", lambda x: x.strip() == "RedistributeAndRefineMeshBlocks", "R&R"),
        ("UMBT", lambda x: x.strip() == "UpdateMeshBlockTree", "Sync"),
    ]

    return spec


def gen_spec_mesh_init() -> list[SingleSpec]:
    funcs = [
        ["ilb&amr", "LoadBalancingAndAdaptiveMeshRefinement"],
        ["ifd", "Task_FillDerived"],
        ["ilasbb", "Task_LoadAndSendBoundBufs"],
        ["irbb", "Task_ReceiveBoundBufs"],
        ["isib", "Task_SetInternalBoundaries"],
        ["itest", "MPI_Test()"],
        ["ibasbb", "Task_BuildSendBoundBufs"],
        ["ipcfd", "Task_PreCommFillDerived"],
        ["iabc", "Task_ApplyBoundaryConditionsOnCoarseOrFine"],
        # ["ip2c", "PrimToCons"],
        # ["iset", "SetGeometry::Set Cached data, Corn"],
        ["iwait", "MPI_Wait()"],
        # ["iset2", "SetGeometry::Set Cached data, Cent"],
        # ["ipgen", "Phoebus::ProblemGenerator::sedov"],
    ]

    rnr = "RedistributeAndRefineMeshBlocks"
    minit = "Mesh::Initialize"
    fpref = f"{rnr} => {minit}"

    def gen(fn: str, fstr: str) -> SingleSpec:
        fstr = f"{fpref} => {fstr}"
        return (fn, lambda x: x.strip().endswith(fstr), fn)

    def kokkos_func(s: str) -> bool:
        schain = [f.strip() for f in s.split("=>")]
        if len(schain) > 3 and schain[-3] == rnr and schain[-2] == minit:
            return "Kokkos" in schain[-1]

        return False

    spec = [gen(*f) for f in funcs]

    totspec: SingleSpec = ("_total", lambda x: x.strip().endswith(fpref), "total")
    spec.append(totspec)

    kspec: SingleSpec = ("ikok", kokkos_func, "ikok")
    spec.append(kspec)

    return spec


def stacked_interactive(trace_key: str):
    suite = TraceUtils.get_traces(trace_key)
    spec = gen_spec_mesh_init()
    spec_keys = list(zip(*spec))[0]

    plot_spec: StackedPlotSpec = {
        "trace_key": trace_key,
        "nranks": suite.nranks,
        "keys": spec_keys,
        "data": {},
        "ylim_top": 50,
        "plot_fname": f"lb_app.meshinit.{trace_key}",
    }

    plot_spec_stacked_inner(suite.traces[0], spec, plot_spec)


def run_plot_steps(trace_key: str):
    plot_fname = "lb_steps.stacked"
    pprof_spec = gen_spec_lb_steps()
    plot_spec: StackedPlotSpec = {
        "trace_key": trace_key,
        "nranks": 1,
        "keys": list(zip(*pprof_spec))[0],
        "data": {},
        "ylim_top": -1,
        "plot_fname": plot_fname,
    }

    suite = TraceUtils.get_traces(trace_key)
    for trace in suite.traces:
        plot_spec_stacked_inner(trace, pprof_spec, plot_spec)


def run_plot_stacked(trace_key: str):
    suite = TraceUtils.get_traces(trace_key)

    all_ylim_top = {
        "mpidbg2048ub22": 1800,
        "mpidbg4096on2048ub22": 2500,
        "mpidbg4096ub22": 1800,
    }

    trace_ylim_top = all_ylim_top[trace_key] if trace_key in all_ylim_top else 2500

    stacked_spec: list[SingleSpec] = [
        ("_Total", lambda x: x.strip() == "Driver_Main", "App"),
        ("C2P::S", lambda x: x.strip() == "ConToPrim::Solve", "Comp"),
        ("CalcFlx", lambda x: x.strip() == "CalculateFluxes", "Comp"),
        ("WSD", lambda x: x.strip() == "WeightedSumData", "Comp"),
        ("SendBB", lambda x: x.strip() == "Task_LoadAndSendBoundBufs", "P2P"),
        ("RecvBB", lambda x: x.strip() == "Task_ReceiveBoundBufs", "P2P"),
        ("R&RMB", lambda x: x.strip() == "RedistributeAndRefineMeshBlocks", "R&R"),
        ("UMBT", lambda x: x.strip() == "UpdateMeshBlockTree", "Sync"),
    ]

    plot_spec: StackedPlotSpec = {
        "trace_key": trace_key,
        "nranks": suite.nranks,
        "keys": list(zip(*stacked_spec))[0],
        "data": {},
        "ylim_top": trace_ylim_top,
        "plot_fname": f"lb_app.stacked.{trace_key}",
    }

    for t in suite.traces:
        plot_spec_stacked_inner(t, stacked_spec, plot_spec)


def run_plot_bar(trace_key: str):
    trace_key = "mpidbg4096ub22.05"
    suite = TraceUtils.get_traces(trace_key)

    spec = gen_spec_std()
    spec = gen_spec_lb_instr()
    spec = gen_spec_mesh_init()
    spec_keys = list(zip(*spec))[2]

    bpspec = gen_bar_plot_spec(trace_key, spec_keys, spec)
    plot_bar_plot_spec(bpspec)

    plot_spec: StackedPlotSpec = {
        "trace_key": trace_key,
        "nranks": suite.nranks,
        "keys": spec_keys,
        "data": {},
        "ylim_top": 50,
        "plot_fname": f"lb_app.meshinit.{trace_key}",
    }

    plot_spec_stacked_inner(suite.traces[0], [spec[2]], plot_spec)
    plot_spec_stacked_inner(suite.traces[3], [spec[2]], plot_spec)

    for t in suite.traces:
        plot_spec_stacked_inner(t, spec, plot_spec)
        break

    policy = "cdpc512par8"
    policy = "hybrid25"
    policy = "lpt"
    phase = "irbb"
    phase = "ifd"
    phase = "ilasbb"
    phase = "SendBB"
    for t in suite.traces:
        policy = t.name
        key = f"{trace_key}_{policy}.{phase}"
        ksum = sum(PPROF_CACHE[key]) / (4096 * 1e6)
        print(f"- {policy:14s}: {ksum:.2f} s")

    sum(PPROF_CACHE["mpidbg4096ub22.04_cdpc512par8.irbb"]) / (4096 * 1e6)
    sum(PPROF_CACHE["mpidbg4096ub22.04_cdpc512par8.irbb"]) / (4096 * 1e6)
    max(PPROF_CACHE["mpidbg4096ub22.04_hybrid25.UMBT"])
    del PPROF_CACHE["mpidbg4096ub22.04_cdpc512par8.isib"]


def run():
    trace_key = "mpidbg2048ub22"
    trace_key = "mpidbg4096on2048ub22"
    run_plot_stacked(trace_key)
    trace_key = "mpidbg4096ub22.04"
    run_plot_bar(trace_key)


if __name__ == "__main__":
    #  global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    #  plot_init()
    plot_init_big()
    run()
