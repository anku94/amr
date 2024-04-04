import glob
import multiprocessing
import numpy as np
import pandas as pd
import io
import ipdb
import pickle
import subprocess
import string
import sys
import time

#  import ray
import re
import traceback
from common import plot_init, plot_init_big, PlotSaver, profile_label_map

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#  from pathlib import path

#  from task import task
from trace_reader import TraceOps

global trace_dir_fmt
trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"


def read_pprof(fpath: str):
    f = open(fpath).readlines()
    lines = [l.strip("\n") for l in f if l[0] != "#"]

    nfuncs = int(re.findall("(\d+)", lines[0])[0])
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
    df = pd.read_csv(
        io.StringIO("\n".join(rel_lines)), delim_whitespace=True, names=prof_cols
    )

    rank = re.findall(r"profile\.(\d+)\.0.0$", fpath)[0]
    df["rank"] = rank
    return df


def dedup_events(all_evts: list):
    all_evts = sorted(all_evts, key=lambda x: len(x), reverse=True)
    dedup_evts = []

    has_evt = lambda ls, x: any([x in lsx for lsx in ls])
    for evt in all_evts:
        if not has_evt(dedup_evts, evt):
            dedup_evts.append(evt)

    print(f"[DedupEvts] Orig: {len(all_evts)}, New: {len(dedup_evts)}")
    return dedup_evts


def remove_events_suffix(all_evts: list, all_suffixes: list):
    filt_evts = []
    aug_sfxs = [s for s in all_suffixes]
    aug_sfxs += [s + " [THROTTLED]" for s in all_suffixes]

    for evt in all_evts:
        if not any([evt.endswith(s) for s in aug_sfxs]):
            filt_evts.append(evt)

    print(f"[FiltEvts] Orig: {len(all_evts)}, New: {len(filt_evts)}")
    return filt_evts


def trim_and_filter_events(events: list):
    trimmed = []

    for e in events:
        e = re.sub(r"(=> )?\[CONTEXT\].*?(?==>|$)", "", e)
        e = re.sub(r"(=> )?\[UNWIND\].*?(?==>|$)", "", e)
        e = re.sub(r"(=> )?\[SAMPLE\].*?(?==>|$)", "", e)
        e = e.strip()

        trimmed.append(e)

    trimmed_uniq = list(set(trimmed))
    trimmed_uniq = [e for e in trimmed_uniq if e != ""]
    trimmed_uniq

    events_mss = sorted(
        [e for e in trimmed_uniq if e.startswith("Multi") and "=>" in e]
    )

    events_driver = [
        e
        for e in trimmed_uniq
        if e.startswith("Driver_Main") and "=>" in e and "Multi" not in e
    ]

    all_events = events_mss + events_driver
    all_events = [e for e in all_events if "Sync" not in e]

    all_events += [".TAU application"]

    print(
        f"Events timmed and dedup'ed. Before: {len(events)}. After: {len(all_events)}"
    )

    print("Events retained: ")
    for e in all_events:
        print(f"\t- {e}")

    #  input("Press ENTER to plot.")

    return all_events


"""
Returns all unique leaf events in the set
Discards intermediate events if leaf events present
"""

def get_top_events(df: pd.DataFrame, cutoff: float = 0.05):
    total_runtime = df[df["name"] == ".TAU application"]["incl_usec"].iloc[0]

    top_df = df[df["incl_usec"] >= total_runtime * cutoff]
    top_names = top_df["name"].unique()

    return trim_and_filter_events(top_names)
    top_names

    cand_names = top_names
    filt_suffixes = [
        "taupreload_main",
        "MPI_Isend()",
        "MPI_Iprobe()",
        "MPI Collective Sync",
    ]
    cand_names = remove_events_suffix(cand_names, filt_suffixes)

    filt_suffixes = [
        ".TAU application => Driver_Main",
        "Driver_Main => MultiStage_Step",
    ]
    cand_names = remove_events_suffix(cand_names, filt_suffixes)
    cand_names = dedup_events(cand_names)

    # remove super long events
    #  all_rel_evts = [ e for e in cand_names if len(e) < 160 ]
    #  print(f"Dropped long events. Orig: {len(cand_names)}, New: {len(all_rel_evts)}")
    all_rel_evts = cand_names

    """ Old athenapk events - may be relevant """
    #  prim_evt = ".TAU application"
    #  prim_df = df[df["name"].str.contains(prim_evt)]
    #  top_df = prim_df[prim_df["incl_usec"] >= total_runtime * cutoff]
    #  all_evts = top_df["name"].to_list()

    #  all_evts = sorted(all_evts, key=lambda x: len(x), reverse=True)
    #  all_rel_evts = []
    #  has_prefix = lambda ls, x: any([lsx.startswith(x) for lsx in ls])
    #  for evt in all_evts:
    #  if not has_prefix(all_rel_evts, evt):
    #  all_rel_evts.append(evt)

    return all_rel_evts


def fold_cam_case(name, cpref=1, csuf=4, sufidx=-2):
    splitted = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", name)).split()

    pref = "".join([s[0:cpref] for s in splitted[:sufidx]])
    suf = "".join([s[0:csuf] for s in splitted[sufidx:]])
    folded_str = pref + suf
    return folded_str


def abbrev_evt(evt: str):
    evt_ls = evt.split("=>")
    evt_ls = evt_ls[-2:]
    evt_clean = []

    for evt_idx, evt in enumerate(evt_ls):
        evt = re.sub(r"Kokkos::[^ ]*", "", evt)
        evt = re.sub(r"Task_", "", evt)
        evt = re.sub(r".*?::", "", evt)
        evt = re.sub(r"\[.*?\]", "", evt)
        evt = evt.strip()
        evt = re.sub(r" ", "_", evt)
        if evt_idx == len(evt_ls) - 1:
            evt_try = fold_cam_case(evt, cpref=1, csuf=4, sufidx=-2)
        else:
            evt_try = fold_cam_case(evt, cpref=1, csuf=1, sufidx=-2)
            #  evt_try = "".join([c for c in evt if c in string.ascii_uppercase])
        if len(evt_try) > 1:
            evt_clean.append(evt_try)
        else:
            evt_clean.append(evt)

    abbrev = "_".join(evt_clean)

    if "MPIA" in abbrev:
        abbrev = "MPI-AllGath"

    return abbrev


def get_event_array(concat_df: pd.DataFrame, event: str) -> list:
    nranks = 512

    ev1 = event
    ev2 = f"{ev1} [THROTTLED]"

    ev1_mask = concat_df["name"] == ev1
    ev2_mask = concat_df["name"] == ev2
    temp_df = concat_df[ev1_mask | ev2_mask]

    #  temp_df = concat_df[concat_df["name"] == event]
    if len(temp_df) != nranks:
        print(
            f"WARN: {event} missing some ranks (nranks={nranks}), found {len(temp_df)}"
        )
    else:
        return temp_df["incl_usec"].to_numpy()
        pass

    all_rank_data = []
    all_ranks_present = temp_df["rank"].to_list()

    temp_df = temp_df[["incl_usec", "rank"]].copy()
    join_df = pd.DataFrame()
    join_df["rank"] = range(nranks)
    join_df = join_df.merge(temp_df, how="left").fillna(0).astype({"incl_usec": int})
    data = join_df["incl_usec"].to_numpy()
    return data


def filter_relevant_events(concat_df: pd.DataFrame, events: list[str]):
    temp_df = concat_df[concat_df["rank"] == 0].copy()
    temp_df.sort_values(["incl_usec"], inplace=True, ascending=False)

    all_data = {}

    for event in events:
        all_data[event] = get_event_array(concat_df, event)

    return all_data


def read_all_pprof_simple(trace_dir: str):
    pprof_glob = f"{trace_dir}/profile/profile.*"
    #  pprof_files = list(map(lambda x: f"{trace_dir}/profile/profile.{x}.0.0", range(32)))
    all_files = glob.glob(pprof_glob)
    #  all_files = pprof_files

    print(f"Trace dir: {trace_dir}, reading {len(all_files)} files")

    with multiprocessing.Pool(16) as pool:
        all_dfs = pool.map(read_pprof, all_files)

    concat_df = pd.concat(all_dfs)
    concat_df["rank"] = concat_df["rank"].astype(int)
    concat_df.sort_values(["rank"], inplace=True)

    concat_df["name"] = concat_df["name"].str.strip()
    return concat_df


def read_all_pprof(trace_dir: str, events: list[str]):
    pprof_glob = f"{trace_dir}/profile/profile.*"
    all_files = glob.glob(pprof_glob)

    with multiprocessing.Pool(16) as pool:
        all_dfs = pool.map(read_pprof, all_files)

    concat_df = pd.concat(all_dfs)
    concat_df["rank"] = concat_df["rank"].astype(int)
    concat_df.sort_values(["rank"], inplace=True)

    concat_df["name"] = concat_df["name"].str.strip()

    key_tot = ".TAU application"

    pprof_data = filter_relevant_events(concat_df, events + [key_tot])
    return pprof_data


def setup_plot_stacked_ph_old():
    stack_keys = [
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_SendBoundaryBuffers_MeshData",
        "Task_ReceiveBoundaryBuffers_MeshData",
        ".TAU application",
    ]

    stack_labels = [
        "$FD_{CO}$",
        "$CF_{CN}$",
        "$AG_{NO}$",
        "$RR_{NO}$",
        "$BC\_SND_{NO}$",
        "$BC\_RCV_{NO}$",
    ]

    ylim = 11000
    ymaj = 1000
    ymin = 200


def setup_plot_stacked_ph_new():
    stack_keys = [
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_LoadAndSendBoundBufs",
        "Task_ReceiveBoundBufs",
        ".TAU application",
    ]

    stack_labels = [
        "$FD_{CO}$",
        "$CF_{CN}$",
        "$AG_{NO}$",
        "$RR_{NO}$",
        "$BC\_SND_{NO}$",
        "$BC\_RCV_{NO}$",
    ]

    ylim = 11000
    ymaj = 1000
    ymin = 200


def setup_plot_stacked_par_vibe():
    stack_keys = [
        "Task_FillDerived",
        "Task_burgers_CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_LoadAndSendBoundBufs",
        "Task_ReceiveBoundBufs",
    ]

    stack_keys_extra = [
        "MPI_Allreduce()",
        "MultiStage_Step => Task_SetInternalBoundaries",
        "MultiStage_Step => Task_WeightedSumData",
        "MultiStage_Step => FluxDivergenceMesh",
    ]

    stack_labels = [
        "$MSS\_FD_{CO}$",
        "$MSS\_CF_{CN}$",
        "$LB\_AG_{NO}$",
        "$LB\_RR_{NO}$",
        "$BC\_SND_{NO}$",
        "$BC\_RCV_{NO}$",
    ]

    stack_labels_extra = [
        "$AR_{NO}$",
        "$MSS\_SIB_{CO}$",
        "$MSS\_WSD_{CO}$",
        "$MSS\_FDM_{CO}$",
    ]

    #  stack_keys += stack_keys_extra
    #  stack_labels += stack_labels_extra

    ylim = 1000
    ymaj = ylim / 10
    ymin = ymaj / 5


def setup_plot_stacked_generic(trace_name):
    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_name)

    concat_df = read_all_pprof_simple(trace_dir)
    stack_keys = get_top_events(concat_df, 0.02)
    stack_labels = list(map(abbrev_evt, stack_keys))

    key_tot = ".TAU application"
    stack_labels[stack_keys.index(key_tot)] = ""

    stack_keys
    stack_labels

    drop_idxes = [
        i
        for i in range(len(stack_labels))
        if len(stack_labels[i]) > 20 or "/" in stack_labels[i]
    ]
    stack_keys = [stack_keys[i] for i in range(len(stack_keys)) if i not in drop_idxes]
    stack_labels = [
        stack_labels[i] for i in range(len(stack_labels)) if i not in drop_idxes
    ]

    ylim = 18000 / 5
    ymaj = 2000 / 5
    ymin = 500 / 5

    ylim, ymaj, ymin = 12000, 1000, 200

    ylim, ymaj, ymin = 3500, 500, 100
    #  ylim, ymaj, ymin = 1500, 250, 50
    #  ylim, ymaj, ymin = 1000, 200, 40

    return stack_keys, stack_labels, ylim, ymaj, ymin


"""
Setup stack_keys, stack_labels, ylim, ymin, ymaj before calling this
"""


def plot_stacked_pprof(trace_name: str):
    print(f"Running plot_stacked_pprof for trace: {trace_name}")

    setup_tuple = setup_plot_stacked_generic(trace_name)
    stack_keys, stack_labels, ylim, ymaj, ymin = setup_tuple

    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_name)
    trace_label = profile_label_map[trace_name]

    concat_df = read_all_pprof_simple(trace_dir)
    pprof_data = filter_relevant_events(concat_df, stack_keys)

    data_y = [pprof_data[k] for k in stack_keys if k != ".TAU application"]
    data_x = np.arange(len(data_y[0]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.stackplot(data_x, *data_y, labels=stack_labels, zorder=2)

    data_y_app = pprof_data[".TAU application"]
    y_mean = int(np.mean(data_y_app) / 1e6)
    print(f"Total mean: {y_mean} s")

    # AGGR PLOT
    ax.plot(data_x, data_y_app, label="APP", zorder=2, linewidth=3)

    ax.set_title(f"Runtime Breakdown - {trace_label}")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Time (s)")

    ax.set_ylim([0, ylim * 1e6])
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ymaj * 1e6))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(ymin * 1e6))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    #  ax.legend(ncol=4)
    ax.legend(ncol=5, fontsize=8)

    fig.tight_layout()
    plot_fname = f"runtime.pprof.stacked.rankwise.{trace_name}"
    PlotSaver.save(fig, "", None, plot_fname)

    # INDVL PLOTS
    for lab, dy in zip(stack_labels, data_y):
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.plot(data_x, dy, label=lab, zorder=2)
        ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
        ax.set_ylim(bottom=0)
        ax.set_title(f"Pprof Component: {lab}")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(64))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(16))
        plt.grid(visible=True, which="major", color="#999", zorder=0)
        plt.grid(visible=True, which="minor", color="#ddd", zorder=0)
        plot_fname = f"runtime.pprof.stacked.com{lab.lower()}.{trace_name}"
        PlotSaver.save(fig, "", None, plot_fname)


def interpolate_data(data: np.array):
    valid_data = [d for d in data if d != 0]
    valid_mean = np.mean(valid_data)
    interpolated_data = [valid_mean if d == 0 else d for d in data]
    return np.array(interpolated_data)


def get_rankhours(trace_dir: str, keys: list[str]) -> dict:
    keys_to_interpolate = ["Task_ReceiveBoundaryBuffers_MeshData"]

    pprof_data = read_all_pprof(trace_dir, keys)

    rh_data = {}
    for k in keys:
        if k not in pprof_data:
            print(f"WARN - rankhour key ({k}) not in pprof data")
            continue

        if k in keys_to_interpolate:
            print(f"Interpolating rankhours for key {k}")
            data = interpolate_data(pprof_data[k])
        else:
            data = pprof_data[k]

        sum_rsec = data.sum() / 1e6
        sum_rh = sum_rsec / 3600
        rh_data[k] = sum_rh

    return rh_data


def plot_rankhour_comparison(trace_dirs: list[str]):
    all_keys = [
        ".TAU application",
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_SendBoundaryBuffers_MeshData",
        "Task_ReceiveBoundaryBuffers_MeshData",
    ]

    all_keys = [
        ".TAU application",
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_LoadAndSendBoundBufs",
        "Task_ReceiveBoundBufs",
    ]

    all_labels = [
        "APP",
        "$FD_{CO}$",
        "$CF_{CN}$",
        "$AG_{NO}$",
        "$RR_{NO}$",
        "$BC\_SND_{NO}$",
        "$BC\_RCV_{NO}$",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    data_x = np.arange(len(all_keys))
    labels_x = all_labels

    all_data_x = []
    all_prof_labels = []
    all_labels = []

    total_width = 0.7
    nbins = len(trace_dirs)
    bin_width = total_width / nbins

    for bidx, trace_dir in enumerate(trace_dirs):
        times = get_rankhours(trace_dir, all_keys)

        label = trace_dir.split("/")[-1]
        all_labels.append(label)

        prof_label = profile_label_map[label]
        all_prof_labels.append(prof_label)

        # assert all keys appear in same order
        data_xt = times.keys()
        all_data_x.append(data_xt)
        for xi in list(zip(*all_data_x)):
            assert len(set(xi)) == 1

        data_y = [times[k] for k in all_keys]
        data_x_bin = data_x + bidx * bin_width

        p = ax.bar(data_x_bin, data_y, bin_width, label=prof_label, zorder=2)
        ax.bar_label(
            p, fmt=lambda x: int(x * 7.03), rotation="vertical", fontsize=10, padding=4
        )

    ax.set_xlabel("Phase Name")
    ax.set_ylabel("Rank-Hours (512 ranks * hrs/rank)")

    prof_title = " vs ".join(all_prof_labels)
    ax.set_title(f"Phase-wise Time Comparison")

    ax.set_xticks(data_x, labels_x)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 1500])

    ax.legend()

    fig.tight_layout()

    #  label_fname = "_".join(all_labels)
    label_fname = get_fname(all_labels)
    fname = f"pprof_rh_{label_fname}"
    PlotSaver.save(fig, "", None, fname)


def plot_rankhour_comparison_new_util(
    ax, all_keys, data_x_bin, data_y, plot_label: bool
):
    bars = [
        {"name": "Total", "ids": [0]},
        {"name": "Compute", "ids": [1, 2]},
        {"name": "LBandAMR", "ids": [3, 4]},
        {"name": "BoundaryComm", "ids": [5, 6]},
    ]

    total_width = 0.7
    nbins = len(bars)
    bin_width = total_width / nbins

    for bar_id, bspec in enumerate(bars):
        bname = bspec["name"]
        bids = bspec["ids"]

        bottom = 0
        for did in bids:
            dx = data_x_bin[bar_id]
            dy = data_y[did]
            #  dlabel = all_keys[did]
            dlabel = ""

            #  if not plot_label:
            #  dlabel = ""

            plot_bars = ax.bar(
                [dx],
                [dy],
                bottom=bottom,
                width=bin_width * 0.6,
                label=dlabel,
                color=f"C{did}",
                zorder=2,
            )

            bottom += dy

            if did == bids[-1]:
                bar_labels = [f"{int(x)} s" for x in [bottom * 7]]
                ax.bar_label(plot_bars, bar_labels, padding=2, fontsize=14, rotation=90)

    labels_x = [x["name"] for x in bars]
    ax.set_xticks(np.arange(len(labels_x)), labels_x)
    pass


def plot_rankhour_comparison_new(trace_dirs: list[str]):
    all_keys = [
        ".TAU application",
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_SendBoundaryBuffers_MeshData",
        "Task_ReceiveBoundaryBuffers_MeshData",
    ]

    all_keys = [
        ".TAU application",
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_LoadAndSendBoundBufs",
        "Task_ReceiveBoundBufs",
    ]

    all_labels = [
        "APP",
        "$FD_{CO}$",
        "$CF_{CN}$",
        "$AG_{NO}$",
        "$RR_{NO}$",
        "$BC\_SND_{NO}$",
        "$BC\_RCV_{NO}$",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    data_x = np.arange(len(all_keys))
    labels_x = all_labels

    all_data_x = []
    all_prof_labels = []
    all_labels = []

    total_width = 0.7
    nbins = len(trace_dirs)
    bin_width = total_width / nbins

    all_trace_labels = [
        "1. Baseline",
        "2. LPT",
        "3. Contig-Improved",
        "4. COntig-Improved-Iterative",
        "5. CI-Iter + ManualLB",
    ]

    for bidx, trace_dir in enumerate(trace_dirs):
        times = get_rankhours(trace_dir, all_keys)

        label = trace_dir.split("/")[-1]
        all_labels.append(label)

        prof_label = profile_label_map[label]
        all_prof_labels.append(prof_label)

        # assert all keys appear in same order
        data_xt = times.keys()
        all_data_x.append(data_xt)
        for xi in list(zip(*all_data_x)):
            assert len(set(xi)) == 1

        data_y = [times[k] for k in all_keys]
        data_x_bin = data_x + bidx * bin_width

        #  p = ax.bar(data_x_bin, data_y, bin_width, label=prof_label, zorder=2)
        #  ax.bar_label(
        #  p, fmt=lambda x: int(x * 7.03), rotation="vertical", fontsize=10, padding=4
        #  )

        plot_label = True if bidx == 0 else False

        plot_rankhour_comparison_new_util(
            ax, all_keys, data_x_bin, data_y, plot_label=plot_label
        )
        ax.plot([], [], label=all_trace_labels[bidx], color="black")

    ax.set_xlabel("Phase Name")
    ax.set_ylabel("Rank-Hours (512 ranks * hrs/rank)")

    prof_title = " vs ".join(all_prof_labels)
    ax.set_title(f"Phase-wise Time Comparison")

    #  ax.set_xticks(data_x, labels_x)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 1600])

    ax.legend()

    fig.tight_layout()

    #  label_fname = "_".join(all_labels)
    label_fname = get_fname(all_labels)
    fname = f"pprof_rh_new_{label_fname}"
    PlotSaver.save(fig, "", None, fname)


def get_summary_simplified(trace_idx):
    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_idx)

    all_keys = [
        ".TAU application",
        "Task_FillDerived",
        "CalculateFluxes",
        "UpdateMeshBlockTree",
        "RedistributeAndRefineMeshBlocks",
        "Task_SendBoundaryBuffers_MeshData",
        "Task_ReceiveBoundaryBuffers_MeshData",
    ]

    times = get_rankhours(trace_dir, all_keys)
    tvals = [times[k] for k in all_keys]
    t_unaccounted = tvals[0] - sum(tvals[1:])

    t_comp = tvals[1] + tvals[2] + tvals[4] + t_unaccounted
    t_sync = tvals[3]
    t_comm = tvals[5] + tvals[6]

    tsumm = {"comp": t_comp, "comm": t_comm, "sync": t_sync}

    return tsumm


def get_summary_simplified_mean(traces: list[int]):
    props = ["comp", "comm", "sync"]
    prop_vals = []

    for trace_idx in traces:
        times_t = get_summary_simplified(trace_idx)
        vals_t = [times_t[p] for p in props]
        print(f"Trace {trace_idx}: vals {vals_t}")
        prop_vals.append(vals_t)

    mean_vals = np.array(prop_vals).mean(axis=0)
    return props, mean_vals


def get_fname(lst):
    if not lst:
        return "", []
        pass

    # Find the shortest string
    shortest_str = min(lst, key=len)

    # Find common prefix
    common_prefix = ""
    for i in range(len(shortest_str)):
        if all(string[i] == shortest_str[i] for string in lst):
            common_prefix += shortest_str[i]
        else:
            break

    # Find residuals
    residuals = [string[len(common_prefix) :] for string in lst]

    res_str = "_".join(residuals)
    fname = f"{common_prefix}_{res_str}"
    print(f"File Name: {fname}")
    return fname


def plot_summary_simplified():
    props, tsumm_baseline = get_summary_simplified_mean([31, 34])
    props, tsumm_lpt = get_summary_simplified_mean([32, 35])
    props, tsumm_con_imp = get_summary_simplified_mean([33, 36])

    data_x = np.arange(3)
    labels_x = [
        "Baseline\n(Contiguous Policy)",
        "Longest Processing\nTime Policy",
        "Contiguous-Improved\nPolicy",
    ]

    tsumm = [tsumm_baseline, tsumm_lpt, tsumm_con_imp]
    tsumm_t = np.array(list(zip(*tsumm)))
    tsumm_t = tsumm_t * 3600 / 512.0

    phases = ["Computation", "Communication", "Synchronization"]
    phases = ["Constant Work", "Communication Overhead", "Synchronization Overhead"]
    order = [0, 1, 2]

    phases = [phases[o] for o in order]
    all_data = [tsumm_t[o] for o in order]

    fig, ax = plt.subplots(1, 1, figsize=(8, 9))
    width = 0.5

    bottom = np.zeros(tsumm_t.shape[1])
    for data_y, phase in zip(all_data, phases):
        bars = ax.bar(data_x, data_y, width=width, bottom=bottom, label=phase, zorder=2)
        bottom += data_y

    bar_labels = [f"{int(x)} s" for x in bottom]
    ax.bar_label(bars, bar_labels, padding=2, fontsize=14)

    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.01))
    ax.set_xticks(data_x, labels_x)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(200))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 10300])

    ax.set_title("Policy-Wise Breakdown of AMR Perf")
    ax.set_ylabel("Time (s)")

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x))
    fig.tight_layout()

    plot_fname = "lb.summary.simpl"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def run_plot_stacked():
    traces = [10, 22, 29, 30, 31, 32, 33]
    traces = [37, 38, 39]
    traces = [40, 41, 42, 43, 44]
    traces = [44]

    for trace_idx in traces:
        trace_name = f"profile{trace_idx}"
        #  plot_stacked_pprof_ph_new(trace_name)

    #  plot_stacked_pprof_par_vibe("burgers1")
    trace_name = "athenapk5"
    trace_name = "athenapk14"
    trace_name = "athenapk15"
    trace_name = "athenapk16"

    trace_name = "profile40"
    trace_name = "stochsg2"
    trace_dir = trace_dir_fmt.format(trace_name)
    concat_df = read_all_pprof_simple(trace_dir)
    get_top_events(concat_df)
    #  setup_plot_stacked_generic(trace_name)
    plot_stacked_pprof(trace_name)


def run_plot_bar():
    global trace_dir_fmt

    traces = ["athenapk14", "athenapk15"]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    trace_dirs

    traces = [37, 38, 39]
    traces = [40, 41, 42, 43, 44]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(f"profile{x}"), traces))
    plot_rankhour_comparison(trace_dirs)
    return

    traces = [31, 34, 32, 35, 33, 36]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)

    traces = [34, 35, 36]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)

    traces = [31, 32, 33]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)

    traces = [22, 28, 30]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)

    traces = [10, 22, 31]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)

    traces = [28, 32]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)

    traces = [30, 33]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    plot_rankhour_comparison(trace_dirs)


def run():
    run_plot_stacked()
    #  run_plot_bar()
    #  plot_summary_simplified()


if __name__ == "__main__":
    #  global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    #  plot_init()
    plot_init_big()
    run()
