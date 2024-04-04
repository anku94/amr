import matplotlib

import os
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
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path

#  from task import Task
from trace_reader import TraceOps
from typing import List, Dict

global trace_dir_fmt
trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"

"""
Returns all unique leaf events in the set
Discards intermediate events if leaf events present
"""

def fold_cam_case(name, cpref=1, csuf=4, sufidx=-2):
    splitted = re.sub("([A-Z][a-z]+)", r" \1",
                      re.sub("([A-Z]+)", r" \1", name)).split()

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


def get_event_array(concat_df: pd.DataFrame, event: str, nranks) -> List:
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
    join_df = join_df.merge(temp_df, how="left").fillna(
        0).astype({"incl_usec": int})
    data = join_df["incl_usec"].to_numpy()
    return data


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


def read_all_pprof_simple(trace_dir: str):
    pprof_glob = f"{trace_dir}/profile/profile.*"
    all_files = glob.glob(pprof_glob)
    #  pprof_files = list(map(lambda x: f"{trace_dir}/profile/profile.{x}.0.0", range(32)))
    #  all_files = pprof_files

    print(f"Trace dir: {trace_dir}, reading {len(all_files)} files")

    with multiprocessing.Pool(16) as pool:
        all_dfs = pool.map(read_pprof, all_files)

    concat_df = pd.concat(all_dfs)
    #  del all_dfs

    concat_df["rank"] = concat_df["rank"].astype(int)
    concat_df.sort_values(["rank"], inplace=True)

    concat_df["name"] = concat_df["name"].str.strip()
    return concat_df


def filter_relevant_events(concat_df: pd.DataFrame, events: List[str], nranks):
    temp_df = concat_df[concat_df["rank"] == 0].copy()
    temp_df.sort_values(["incl_usec"], inplace=True, ascending=False)

    all_data = {}

    for event in events:
        all_data[event] = get_event_array(concat_df, event, nranks)

    return all_data


def read_pprof_cached(trace_dir, stack_keys, nranks):
    pickle_path = f"{trace_dir}/pprof.cache.pickle"
    if os.path.exists(pickle_path):
        return pickle.loads(open(pickle_path, "rb").read())

    concat_df = read_all_pprof_simple(trace_dir)
    pprof_data = filter_relevant_events(concat_df, stack_keys, nranks)
    with open(pickle_path, "wb+") as f:
        f.write(pickle.dumps(pprof_data))

    return pprof_data


def transform_pprof_cached(pprof_data):
    keys = list(pprof_data.keys())
    kmpi = "MPI Collective Sync"
    klb = "LoadBalancingAndAdaptiveMeshRefinement"

    k1 = kmpi
    k2 = f"UpdateMeshBlockTree => {k1}"
    k3 = f"Driver_Main => {klb}"
    k4 = f"{klb} => UpdateMeshBlockTree"
    k5 = f"{klb} => RedistributeAndRefineMeshBlocks"

    keys_tmp = [k1, k2, k3, k4, k5]

    for k in keys_tmp:
        print(f"Checking if {k} in keys")
        assert k in keys

    keys_final = [k for k in keys if k not in keys_tmp and "MPI" not in k]

    data_final = {k: pprof_data[k] for k in keys_final}
    data_final[klb] = pprof_data[k4] + pprof_data[k5] - pprof_data[k2]
    data_final[kmpi] = pprof_data[kmpi]

    return data_final


def preprocess_pprof_data(trace_dirs, nranks):
    trace0_df = read_all_pprof_simple(trace_dirs[0])
    stack_keys = get_top_events(trace0_df, 0.02)
    del trace0_df

    time.sleep(4)

    for t in trace_dirs:
        read_pprof_cached(t, stack_keys, nranks)
        time.sleep(2)


def setup_plot_stacked_generic(trace_name):
    global trace_dir_fmt
    trace_name = "stochsg10"
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
    stack_keys = [stack_keys[i]
                  for i in range(len(stack_keys)) if i not in drop_idxes]
    stack_labels = [
        stack_labels[i] for i in range(len(stack_labels)) if i not in drop_idxes
    ]

    ylim = 18000 / 5
    ymaj = 2000 / 5
    ymin = 500 / 5

    #  ylim, ymaj, ymin = 12000, 1000, 200

    ylim, ymaj, ymin = 3000, 500, 100
    #  ylim, ymaj, ymin = 1500, 250, 50
    #  ylim, ymaj, ymin = 1000, 200, 40

    return stack_keys, stack_labels, ylim, ymaj, ymin


def plot_stacked_pprof(trace_dir: str, trace_label: str, nranks: int):
    trace_name = os.path.basename(trace_dir)
    print(f"Running plot_stacked_pprof for trace: {trace_name}, {trace_label}")

    pprof_data = read_pprof_cached(trace_dir, [], nranks)
    pprof_data = transform_pprof_cached(pprof_data)
    stack_keys = pprof_data.keys()
    stack_keys = [k for k in stack_keys if k != ".TAU application"]
    skidx = list(range(len(stack_keys)))
    skidx[0] = 1
    skidx[1] = 0
    stack_keys = [stack_keys[i] for i in skidx]
    stack_labels = list(map(abbrev_evt, stack_keys))

    data_y = [pprof_data[k] for k in stack_keys if k != ".TAU application"]
    data_x = np.arange(len(data_y[0]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.stackplot(data_x, *data_y, labels=stack_labels, zorder=2)

    data_y_app = pprof_data[".TAU application"]
    y_mean = int(np.mean(data_y_app) / 1e6)
    print(f"Total mean: {y_mean} s")

    ymax = max(data_y_app)
    ylim = int(np.ceil(ymax / 1e9) * 1e9)

    # AGGR PLOT
    ax.plot(data_x, data_y_app, label="APP", zorder=2, linewidth=3)

    ax.set_title(f"Runtime Breakdown - {trace_label}")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Time (s)")

    ax.set_ylim([0, ylim])
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    #  ax.legend(ncol=4)
    ax.legend(ncol=5, fontsize=8)

    fig.tight_layout()
    plot_fname = f"runtime.pprof.stacked.rankwise.{trace_name}"
    PlotSaver.save(fig, "", None, plot_fname)

    return

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


def trim_and_filter_events(events: List):
    trimmed = []

    for e in events:
        e = re.sub(r"(=> )?\[CONTEXT\].*?(?==>|$)", "", e)
        e = re.sub(r"(=> )?\[UNWIND\].*?(?==>|$)", "", e)
        e = re.sub(r"(=> )?\[SAMPLE\].*?(?==>|$)", "", e)
        e = e.strip()

        trimmed.append(e)

    trimmed_uniq = list(set(trimmed))
    trimmed_uniq = [e for e in trimmed_uniq if e != ""]

    events_mpi = [e for e in trimmed_uniq if "MPI" in e and "=>" not in e]
    events_mpi = [e for e in trimmed_uniq if "MPI Collective Sync" in e]

    events_mss = [e for e in trimmed_uniq if e.startswith(
        "Multi") and "=>" in e]

    events_driver = [
        e
        for e in trimmed_uniq
        if e.startswith("Driver_Main") and "=>" in e and "Multi" not in e
    ]

    events_lb = [
        e for e in events if e.startswith("LoadBalancingAndAdaptiveMeshRefinement =>")
    ]

    #  events_driver
    #  events_lb

    all_events = events_mpi + events_mss + events_driver + events_lb
    all_events = [e for e in all_events if "MPI_" not in e]

    all_events += [".TAU application"]
    all_events

    print(
        f"Events timmed and dedup'ed. Before: {len(events)}. After: {len(all_events)}"
    )

    print("Events retained: ")
    for e in all_events:
        print(f"\t- {e}")

    #  input("Press ENTER to plot.")

    return all_events


def get_top_events(df: pd.DataFrame, cutoff: float = 0.05):
    total_runtime = df[df["name"] == ".TAU application"]["incl_usec"].iloc[0]

    top_df = df[df["incl_usec"] >= total_runtime * cutoff]
    top_names = top_df["name"].unique()

    return trim_and_filter_events(top_names)


def get_key_classification(keys: list[str]) -> dict[str, str]:
    key_map = {}

    for k in keys:
        print(k)
        if "send" in k.lower():
            print("\t- Classified SEND")
            key_map[k] = "send"
        elif "receive" in k.lower():
            print("\t- Classified RECV")
            key_map[k] = "recv"
        elif "loadbalancing" in k.lower():
            print("\t- Classified LB")
            key_map[k] = "lb"
        elif ".tau application" in k.lower():
            print("\t- Classified APP")
            key_map[k] = "app"
        elif "mpi" in k.lower():
            print("\t- Classified SYNC")
            key_map[k] = "sync"
        else:
            print("\t- Classified Compute")
            key_map[k] = "comp"

    return key_map


def get_rankhour_comparison(trace_dirs: List[str], nranks):
    # nranks used for normalizing into rank-hours
    all_pprof_summ = []

    for trace_dir in trace_dirs:
        #  concat_df = read_all_pprof_simple(trace_dir)
        #  pprof_data = filter_relevant_events(concat_df, stack_keys)
        #  del concat_df
        pprof_data = read_pprof_cached(trace_dir, [], nranks)
        pprof_data = transform_pprof_cached(pprof_data)
        key_map = get_key_classification(pprof_data.keys())

        pprof_summ = {}
        for k in pprof_data.keys():
            k_map = key_map[k]
            k_sum = pprof_data[k].sum()
            if k_map in pprof_summ:
                pprof_summ[k_map].append(k_sum)
            else:
                pprof_summ[k_map] = [k_sum]
        all_pprof_summ.append(pprof_summ)

    all_pprof_summ
    all_pprof_summ[0]

    # phase_data = list of {}s, key = phase, val = list of per-trace per-key vals
    phase_data = []
    for idx, t in enumerate(all_pprof_summ):
        norm_phase_times = {}
        for p in t:
            norm_t = (np.array(t[p]) / (nranks * 1e6)).astype(int)
            norm_phase_times[p] = norm_t
        phase_data.append(norm_phase_times)

    phases = ["app", "comp", "send", "recv", "sync", "lb"]
    phase_events = {}
    for phase in phases:
        cur_phase_events = list(
            [p for p in pprof_data.keys() if key_map[p] == phase])
        phase_events[phase] = cur_phase_events

    phase_data

    # aggregate phase_data further
    keys_aggr = set().union(*phase_data)
    phase_data_aggr = {}
    for k in keys_aggr:
        v = [p[k].sum() for p in phase_data]
        phase_data_aggr[k] = v

    # compute "other" as "app" - rest
    sum_vals = np.sum(np.array(list(phase_data_aggr.values())), axis=0)
    phase_data_aggr["other"] = 2 * np.array(phase_data_aggr["app"]) - sum_vals

    phase_data_aggr
    phases.append("other")

    phases = [ p for p in phases if p in phase_data_aggr.keys() ]
    phases = [ p for p in phases if p in phase_events.keys() and len(phase_events[p]) > 0 ]

    keys_to_del = [ k for k in phase_events if len(phase_events[k]) == 0 ]
    for k in keys_to_del:
        del phase_events[k]

    return (phases, phase_events, phase_data_aggr)


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
            if shortest_str[i] not in string.ascii_letters:
                break
            common_prefix += shortest_str[i]
        else:
            break

    # Find residuals
    residuals = [string[len(common_prefix):] for string in lst]

    res_str = "_".join(residuals)
    fname = f"{common_prefix}_{res_str}"
    print(f"File Name: {fname}")
    return fname


def plot_rankhour_comparison_simple(trace_names: List[str]) -> None:
    n_traces = len(traces)
    width = 0.45

    keys_to_plot = ["comp", "send", "recv", "sync", "lb"]
    keys_to_plot = ["comp", "sync", "send", "recv", "lb", "other"]
    key_labels = ["Compute", "Global Barrier",
                  "MPI Send", "MPI Recv", "LoadBalancing", "Other"]
    bottom = np.zeros(n_traces)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    for idx, k in enumerate(keys_to_plot):
        data_x = np.arange(n_traces)
        data_y = phase_data_summ[k]

        label = k[0].upper() + k[1:]
        label = key_labels[idx]
        ax.bar(data_x, data_y, bottom=bottom,
               zorder=2, width=width, label=label)
        bottom += data_y

    p = ax.bar(
        data_x,
        phase_data_summ["other"],
        bottom=bottom,
        zorder=2,
        width=width,
        label="Other",
        color="#999",
    )

    ax.bar_label(
        p,
        fmt="{:.0f} s",
        rotation="horizontal",
        label=phase_data_summ["app"],
        fontsize=14,
        padding=4,
    )

    ax.set_title(
        "Runtime Evolution in Galaxy Sim (10k timesteps)", fontsize=18)
    ax.set_xticks(data_x)
    ax.set_xticklabels(trace_names)
    ax.set_ylabel("Runtime (s)")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} s".format(x))
    )

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(400))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 16000])

    ax.legend(loc="upper right", ncol=3, fontsize=13)

    fig.tight_layout()

    plot_fname = "pprof_rh_simple"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


#  plot_rankhour_comparison_2(trace_names)


def plot_rankhour_comparison(trace_names: list[str], trace_labels: list[str], all_phase_data) -> None:
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    width = 0.2

    phases, phase_events, phase_data = all_phase_data

    n_phases = np.arange(len(phases))
    n_traces = len(trace_names)

    # Reformatting the code with 4-space indentation
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    for i, trace in enumerate(trace_labels):
        keys_trace = phases
        vals_trace = [phase_data[p][i] for p in phases]
        print(keys_trace)
        print(vals_trace)

        data_x = n_phases + i * width
        print(data_x, vals_trace)
        ax.bar(data_x, vals_trace, width, label=trace,
               edgecolor="black", zorder=2)

    #  for i, trace in enumerate(trace_names):
        #  for j, phase in enumerate(phases):
        #  values = phase_data[phase]
        #  bottom_value = 0  # Initialize bottom_value for stacking
        #  for k, value in enumerate(values):
        #  label = trace if j == 0 and k == 0 else ""
        #  ax.bar(
        #  n_phases[j] + i * width,
        #  value,
        #  width,
        #  label=label,
        #  color=f"C{i}",
        #  hatch=hatches[k % len(hatches)],
        #  bottom=bottom_value,
        #  edgecolor="black",
        #  zorder=2,
        #  )
        #  bottom_value += value  # Update bottom_value for next bar in stack
        #  break

    # Labeling and layout
    ax.set_xticks(n_phases + width * (n_phases - 1) / 2)
    ax.set_xticklabels(phases)
    ax.legend(title="Traces")

    label_fname = get_fname(trace_names)
    fname = f"pprof_rh_{label_fname}"

    #  ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    #  ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    #  ax.set_ylim([0, 12000])

    ymax = max(phase_data["app"])
    ylim = int(np.ceil(ymax / 1000) * 1000)
    ax.set_ylim([0, ylim])

    ax.set_xlabel("App Phase")
    ax.set_ylabel("Phase Time (s)")
    ax.set_title("Phase-Wise Perf Breakdown (nranks=1536, ts=20k)")
    fig.tight_layout()

    PlotSaver.save(fig, "", None, fname)


def run_plot_bar_simple():
    global trace_dir_fmt

    #  traces = ["profile40", "profile41"]
    traces = ["stochsg10", "stochsg11", "stochsg12", "stochsg13"]
    traces = ["stochsg10", "stochsg14", "stochsg15", "stochsg16"]
    traces = ["stochsg17", "stochsg18", "stochsg19", "stochsg20"]
    traces = ["stochsg21", "stochsg22", "stochsg23", "stochsg26"]
    #  traces = ["stochsg28", "stochsg29", "stochsg30", "stochsg31"]
    #  traces = ["stochsg32", "stochsg33", "stochsg34", "stochsg35"]
    #  traces = ["stochsg36", "stochsg37", "stochsg38", "stochsg39"]
    traces = ["stochsg40", "stochsg41", "stochsg42", "stochsg43"]
    #  traces = ["stochsg44", "stochsg45", "stochsg46", "stochsg47"]
    # traces = ["stochsg26"]
    #  traces = ["stochsg7", "stochsg8", "stochsg9"]
    trace_labels = ["Baseline", "LPT", "Contiguous-DP", "CDPP++"]
    trace_labels = ["Baseline", "LPT", "Contiguous-DP", "CDPP++"]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))

    nranks = 1536

    # gen .pickle files, use the first time
    preprocess_pprof_data(trace_dirs, nranks)
    # return
    all_phase_data = get_rankhour_comparison(trace_dirs, nranks)
    all_phase_data
    phase_mat = np.array([all_phase_data[2][k] for k in all_phase_data[1] if k in all_phase_data[2]])
    print(phase_mat)
    #  phase_mat[:, 3] - phase_mat[:, 1]
    plot_rankhour_comparison(traces, trace_labels, all_phase_data)
    #  all_phase_data = get_rankhour_comparison_new(trace_dirs)
    #  pass


def run_plot_bar():
    global trace_dir_fmt

    traces = ["athenapk5", "athenapk14", "athenapk15", "athenapk16"]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    trace_dirs
    #  plot_rankhour_comparison(traces)
    for idx in [0, 1]:
        concat_df = read_all_pprof_simple(trace_dirs[idx])
        concat_df_out = f"{trace_dirs[idx]}/pprof_concat.csv"
        concat_df.to_csv(concat_df_out, index=False)

    for k in phase_data[0]:
        phase_data[0][k] = (phase_data[0][k] * 5).astype(int)
        print(phase_data[0][k])

    plot_rankhour_comparison(trace_names)
    plot_rankhour_comparison_2(trace_names)


def run_plot_stacked():
    global trace_dir_fmt
    nranks = 1536

    trace_names = ["stochsg21", "stochsg22", "stochsg23", "stochsg26"]
    trace_labels = ["Baseline", "LPT", "Contiguous-DP", "CDPP"]

    trace_names = ["stochsg26"]
    trace_labels = ["CDPP-NoLog"]

    trace_names = ["stochsg28", "stochsg29", "stochsg30", "stochsg31"]
    trace_labels = ["Baseline", "LPT", "Contiguous-DP", "CDPP"]

    trace_names = ["stochsg32", "stochsg33", "stochsg34", "stochsg35"]
    # trace_names = ["stochsg36", "stochsg37", "stochsg38", "stochsg39"]
    # trace_names = ["stochsg40", "stochsg41", "stochsg42", "stochsg43"]
    trace_names = ["stochsg44", "stochsg45", "stochsg46", "stochsg47"]

    trace_label_prefix = "StochSG-n1024-"
    trace_label_prefix = "StochSG-n1536-3k-"
    # trace_label_prefix = "StochSG-n2048-5k-"
    trace_label_prefix = "StochSG-n2048-20k-"
    trace_labels = [f"{trace_label_prefix}{t}" for t in trace_labels]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), trace_names))

    for trace_dir, trace_label in zip(trace_dirs, trace_labels):
        plot_stacked_pprof(trace_dir, trace_label, nranks)
    pass


def run():
    run_plot_bar_simple()
    #  run_plot_stacked()


if __name__ == "__main__":
    #  global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    #  plot_init()
    plot_init_big()
    run()
