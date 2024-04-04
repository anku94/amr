import glob
import multiprocessing
import numpy as np
import pandas as pd
import io
import ipdb
import pickle
import subprocess
import string
import os
import sys
import time

#  import ray
import re
import traceback
from common import plot_init, plot_init_big, PlotSaver, profile_label_map

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import itertools

from pathlib import Path

#  from task import task
from trace_reader import TraceOps

global trace_dir_fmt
trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"


def flip(items: list, ncols: int) -> list:
    flipped = itertools.chain(*[items[i::ncols] for i in range(ncols)])
    return list(flipped)


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


def fold_cam_case(name, cpref=1, csuf=4, sufidx=-2):
    splitted = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", name)).split()

    pref = "".join([s[0:cpref] for s in splitted[:sufidx]])
    suf = "".join([s[0:csuf] for s in splitted[sufidx:]])
    folded_str = pref + suf
    return folded_str


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


def read_pprof_cached(trace_dir, stack_keys):
    pickle_path = f"{trace_dir}/pprof.cache.pickle"
    if os.path.exists(pickle_path):
        return pickle.loads(open(pickle_path, "rb").read())
        pass

    concat_df = read_all_pprof_simple(trace_dir)
    pprof_data = filter_relevant_events(concat_df, stack_keys)
    with open(pickle_path, "wb+") as f:
        f.write(pickle.dumps(pprof_data))

    return pprof_data


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

    [e for e in trimmed_uniq if e not in all_events]

    all_events += [".TAU application"]
    all_events += ["UpdateMeshBlockTree => MPI_Allgather()"]
    all_events

    print(
        f"Events timmed and dedup'ed. Before: {len(events)}. After: {len(all_events)}"
    )

    print("Events retained: ")
    for e in all_events:
        print(f"\t- {e}")

    #  input("Press ENTER to plot.")

    return all_events


#  trace_name = "profile40"


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

    #  ylim, ymaj, ymin = 3000, 500, 100
    #  ylim, ymaj, ymin = 1500, 250, 50
    #  ylim, ymaj, ymin = 1000, 200, 40

    return stack_keys, stack_labels, ylim, ymaj, ymin


def read_all_pprof_simple(trace_dir: str):
    pprof_glob = f"{trace_dir}/profile/profile.*"
    #  pprof_files = list(map(lambda x: f"{trace_dir}/profile/profile.{x}.0.0", range(32)))
    all_files = glob.glob(pprof_glob)
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


def preprocess_pprof_data(traces, trace_dirs):
    stack_keys, _, _, _, _ = setup_plot_stacked_generic(traces[0])
    stack_keys
    #  stack_keys += [
    #  "RedistributeAndRefineMeshBlocks",
    #  ]

    #  data = read_pprof_cached(trace_dir, stack_keys)

    #  read_pprof_cached(trace_dirs[0], stack_keys)
    #  time.sleep(5)
    #  read_pprof_cached(trace_dirs[1], stack_keys)
    #  time.sleep(5)
    #  read_pprof_cached(trace_dirs[2], stack_keys)
    for trace_dir in trace_dirs:
        read_pprof_cached(trace_dir, stack_keys)
        time.sleep(2)


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
        elif "redistributeandrefine" in k.lower():
            print("\t- Classified LB")
            key_map[k] = "lb"
        elif "loadbalancingandadaptive" in k.lower():
            print("\t- Classified LB+SYNC")
            key_map[k] = "lb+sync"
        elif ".tau application" in k.lower():
            print("\t- Classified APP")
            key_map[k] = "app"
        elif "mpi_all" in k.lower():
            print("\t- Classified SYNC")
            key_map[k] = "sync"
        elif "mpi" in k.lower():
            print("\t- Classified MPIOTHER")
            key_map[k] = "mpioth"
        elif "makeout" in k.lower():
            print("\t- Classified IO")
            key_map[k] = "io"
        else:
            print("\t- Classified Compute")
            key_map[k] = "comp"

    return key_map


def get_rankhour_comparison_offline(trace_dirs: list[str]):
    all_pprof_summ = []

    for trace_dir in trace_dirs:
        #  concat_df = read_all_pprof_simple(trace_dir)
        #  pprof_data = filter_relevant_events(concat_df, stack_keys)
        #  del concat_df
        pprof_data = read_pprof_cached(trace_dir, [])
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

    phase_data = []
    nranks = 512
    for idx, t in enumerate(all_pprof_summ):
        norm_phase_times = {}
        for p in t:
            norm_t = (np.array(t[p]) / (nranks * 1e6)).astype(int)
            norm_phase_times[p] = norm_t
        phase_data.append(norm_phase_times)

    phases = ["app", "comp", "send", "recv", "sync", "lb"]
    phase_events = {}
    for phase in phases:
        cur_phase_events = list([p for p in pprof_data.keys() if key_map[p] == phase])
        phase_events[phase] = cur_phase_events

    phase_data
    return phase_data


def prep_data_simple(trace_dirs: list[str]):
    phase_data = get_rankhour_comparison_offline(trace_dirs)
    phase_data

    print("ALERT - scaling 0 by 5X")
    for k in phase_data[0]:
        phase_data[0][k] = (phase_data[0][k] * 5).astype(int)
        print(phase_data[0][k])

    run_names = [
        "Baseline",
        "LPT",
        "Contiguous-DP",
        "Contiguous-DP++",
        "Contiguous-DP++-1K",
    ]

    #  run_names = [
    #  "Baseline",
    #  "Manually\nTuned",
    #  "LPT",
    #  "Contiguous-DP++",
    #  ]

    run_names = run_names[: len(trace_dirs)]

    phase_data_summ = {}
    for d in phase_data:
        for k in d:
            vsum = np.sum(d[k])
            if k in phase_data_summ:
                phase_data_summ[k].append(vsum)
            else:
                phase_data_summ[k] = [vsum]

    phase_data_summ

    keys_sum = np.zeros(len(trace_dirs), dtype=int)
    keys_sum

    if "lb" not in phase_data_summ:
        a = np.array(phase_data_summ["lb+sync"])
        b = np.array(phase_data_summ["sync"])
        phase_data_summ["lb"] = a - b

        del phase_data_summ["lb+sync"]

    if "mpioth" in phase_data_summ:
        phase_data_summ["sync"] += np.array(phase_data_summ["mpioth"])
        del phase_data_summ["mpioth"]

    phase_data_summ
    if "io" in phase_data_summ:
        del phase_data_summ["io"]

    keys_to_plot = ["comp", "send", "recv", "sync", "lb", "io"]
    keys_to_plot = ["comp", "send", "recv", "sync", "lb"]
    for k in keys_to_plot:
        keys_sum += phase_data_summ[k]

    v_other = phase_data_summ["app"] - keys_sum
    v_other
    phase_data_summ["other"] = v_other

    print(phase_data_summ)
    G_data_x = np.arange(len(run_names))
    return run_names, phase_data_summ


def prep_data_blastwave():
    global G_plot_title
    global G_plot_ylim
    global G_plot_fname
    global G_data_x
    global trace_dir_fmt

    G_plot_title = "Runtime Evolution in Blast Wave (30k timesteps)"
    G_plot_ylim = 12000
    G_plot_fname = f"pprof_rh_simple_blastwave_new"

    traces = ["profile40", "profile41", "profile42", "profile43"]
    trace_dirs = [trace_dir_fmt.format(t) for t in traces]
    phase_data = get_rankhour_comparison_offline(trace_dirs)

    run_names = [
        "Baseline",
        "LPT",
        "Contiguous-DP",
        "Contiguous-DP++",
        #  "Contiguous-DP++-1K",
    ]

    phase_data_summ = {}
    for d in phase_data:
        for k in d:
            vsum = np.sum(d[k])
            if k in phase_data_summ:
                phase_data_summ[k].append(vsum)
            else:
                phase_data_summ[k] = [vsum]

    #  if "lb" not in phase_data_summ:
    a = np.array(phase_data_summ["lb+sync"])
    b = np.array(phase_data_summ["sync"])
    phase_data_summ["lb"] = a - b

    del phase_data_summ["lb+sync"]

    #  if "mpioth" in phase_data_summ:
    phase_data_summ["sync"] += np.array(phase_data_summ["mpioth"])
    del phase_data_summ["mpioth"]

    #  if "io" in phase_data_summ:
    #  del phase_data_summ["io"]

    keys_sum = np.zeros(len(trace_dirs), dtype=int)
    keys_to_plot = ["comp", "send", "recv", "sync", "lb", "io"]
    for k in keys_to_plot:
        keys_sum += phase_data_summ[k]

    v_other = phase_data_summ["app"] - keys_sum
    phase_data_summ["other"] = v_other

    phase_data_summ

    G_data_x = np.arange(len(run_names))
    # Build some distance between 3 and 4
    if len(G_data_x) >= 4:
        G_data_x = G_data_x.astype(float)
        G_data_x[2] -= 0.06
        G_data_x[3] += 0.06

    return run_names, phase_data_summ


def prep_data_athenapk():
    global G_plot_title
    global G_plot_ylim
    global G_plot_fname
    global G_data_x
    global trace_dir_fmt

    G_plot_title = "Runtime Evolution in Galaxy Cluster (10k timesteps)"
    G_plot_ylim = 16000
    G_plot_fname = f"pprof_rh_simple_glxcul_new"

    traces = ["athenapk5", "athenapk14", "athenapk15", "athenapk16"]
    trace_dirs = [trace_dir_fmt.format(t) for t in traces]

    phase_data = get_rankhour_comparison_offline(trace_dirs)
    phase_data

    print("ALERT - scaling 0 by 5X")
    for k in phase_data[0]:
        phase_data[0][k] = (phase_data[0][k] * 5).astype(int)
        print(phase_data[0][k])

    run_names = [
        "Baseline",
        "Manually\nTuned",
        "LPT",
        "Contiguous-DP++",
    ]

    phase_data_summ = {}
    for d in phase_data:
        for k in d:
            vsum = np.sum(d[k])
            if k in phase_data_summ:
                phase_data_summ[k].append(vsum)
            else:
                phase_data_summ[k] = [vsum]

    keys_sum = np.zeros(len(trace_dirs), dtype=int)
    keys_to_plot = ["comp", "send", "recv", "sync", "lb"]
    for k in keys_to_plot:
        keys_sum += phase_data_summ[k]

    v_other = phase_data_summ["app"] - keys_sum
    phase_data_summ["other"] = v_other

    G_data_x = np.arange(len(run_names))
    return run_names, phase_data_summ


def plot_rankhour_comparison_simple(run_names: list[str], run_data) -> None:
    global G_plot_title
    global G_plot_ylim
    global G_plot_fname
    global G_data_x

    n_traces = len(run_names)
    width = 0.45

    print(f"Plotting for {n_traces} traces.")

    keys_to_plot = ["comp", "send", "recv", "sync", "lb"]
    keys_to_plot = ["comp", "sync", "send", "recv", "lb", "io"]
    keys_to_plot = ["comp", "sync", "send", "recv", "lb", "io"]
    keys_to_plot = ["comp", "io", "send", "recv", "sync", "other", "lb"]

    key_labels = {
        "comp": "Compute",
        "sync": "Global Barrier",
        "send": "MPI Send",
        "recv": "MPI Recv",
        "lb": "LoadBalancing",
        "io": "IO",
        "other": "Other Comm/Sync",
    }

    bottom = np.zeros(n_traces)
    data_x = G_data_x[:n_traces]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    #  ax.set_title(G_plot_title, fontsize=18)

    ax.set_xticks(data_x)
    ax.set_xticklabels(run_names[:n_traces], fontsize=14)
    ax.set_ylabel("Runtime (s)\n(lower is better)", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:,.0f} s".format(x))
    )

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(400))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim([0, G_plot_ylim])

    fig.tight_layout()
    ax.set_xlim([-0.3975, 3.3975])
    fig.subplots_adjust(bottom=0.1, right=0.9)

    for idx, k in enumerate(keys_to_plot):
        data_y = run_data[k][:n_traces]

        label = k[0].upper() + k[1:]
        #  label = key_labels[idx]
        label = key_labels[k]
        p = ax.bar(data_x, data_y, bottom=bottom, zorder=2, width=width, label=label)
        bottom += data_y

    #  p = ax.bar(
    #  data_x,
    #  run_data["other"][:n_traces],
    #  bottom=bottom,
    #  zorder=2,
    #  width=width,
    #  label="Other Comm/Sync",
    #  color="#999",
    #  )

    ax.bar_label(
        p,
        fmt="{:.0f} s",
        rotation="horizontal",
        label=run_data["app"][:n_traces],
        fontsize=14,
        padding=4,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        flip(handles, 3),
        flip(labels, 3),
        loc="upper left",
        ncol=3,
        fontsize=13,
        bbox_to_anchor=(0.00, 1.02),
    )

    #  ax.legend(loc="upper left", ncol=3, fontsize=13, bbox_to_anchor=(0.00, 1.02))

    PlotSaver.save(fig, "", None, f"{G_plot_fname}_{n_traces}")


def run():
    #  run_names, run_data = prep_data_simple(trace_dirs)
    #  run_names, run_data = prep_data_athenapk()
    run_names, run_data = prep_data_blastwave()
    n_traces = list(range(len(run_names), 0, -1))
    #  n_traces = [4]

    for nt in n_traces:
        plot_rankhour_comparison_simple(run_names[:nt], run_data)


if __name__ == "__main__":
    #  global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    #  plot_init()
    plot_init_big()
    run()
