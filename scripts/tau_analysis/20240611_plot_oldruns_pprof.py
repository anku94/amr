import glob
import io
import json
import multiprocessing
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import plot_init_big as plot_init, PlotSaver


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

    events_mpi = [e for e in trimmed_uniq if "MPI" in e and "=>" not in e]
    events_mpi = [e for e in events_mpi if "MPI Collective Sync" in e]

    events_mss = [e for e in trimmed_uniq if e.startswith("Multi") and "=>" in e]

    events_driver = [
        e
        for e in trimmed_uniq
        if e.startswith("Driver_Main")
        and "=>" in e
        and "Multi" not in e
        and "MPI" not in e
    ]

    e_fc = [
            "MultiStage_Step => Task_LoadAndSendFluxCorrections",
            "MultiStage_Step => Task_ReceiveFluxCorrections"
            ]

    events_lb = [
        e for e in events if e.startswith("LoadBalancingAndAdaptiveMeshRefinement =>")
    ]

    #  events_driver
    #  events_lb

    all_events = events_mpi + events_mss + events_driver + events_lb + e_fc
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


def classify_key(k: str) -> str:
    print(k)
    if "send" in k.lower():
        print("\t- Classified SEND")
        return "send"
    elif "receive" in k.lower():
        print("\t- Classified RECV")
        return "recv"
    elif "loadbalancing" in k.lower():
        print("\t- Classified LB")
        return "lb"
    elif ".tau application" in k.lower():
        print("\t- Classified APP")
        return "app"
    elif "mpi" in k.lower():
        print("\t- Classified SYNC")
        return "sync"
    else:
        print("\t- Classified Compute")
        return "comp"


def get_key_classification(keys: list[str]) -> dict[str, str]:
    key_map = {}

    for k in keys:
        key_map[k] = classify_key(k)

    return key_map


def get_event_array(concat_df: pd.DataFrame, event: str, nranks) -> list:
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
    df = pd.read_csv(io.StringIO("\n".join(rel_lines)), sep="\s+", names=prof_cols)

    rank = re.findall(r"profile\.(\d+)\.0.0$", fpath)[0]
    df["rank"] = rank
    return df


def read_all_pprof_simple(trace_dir: str):
    pprof_glob = f"{trace_dir}/profile/profile.*"
    all_files = glob.glob(pprof_glob)
    #  pprof_files = list(map(lambda x: f"{trace_dir}/profile/profile.{x}.0.0", range(32)))
    #  all_files = pprof_files

    print(f"-----\nTrace dir: {trace_dir}, reading {len(all_files)} files")

    with multiprocessing.Pool(16) as pool:
        all_dfs = pool.map(read_pprof, all_files)

    concat_df = pd.concat(all_dfs)
    #  del all_dfs

    concat_df["rank"] = concat_df["rank"].astype(int)
    concat_df.sort_values(["rank"], inplace=True)

    concat_df["name"] = concat_df["name"].str.strip()
    return concat_df


def filter_relevant_events(concat_df: pd.DataFrame, events: list[str], nranks):
    temp_df = concat_df[concat_df["rank"] == 0].copy()
    temp_df.sort_values(["incl_usec"], inplace=True, ascending=False)

    all_data = {}

    for event in events:
        all_data[event] = get_event_array(concat_df, event, nranks)

    return all_data


def run_get_straggler_dict(run: str) -> dict:
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    trace_dir = trace_dir_fmt.format(run)

    concat_df = read_all_pprof_simple(trace_dir)
    concat_df

    straggler_events = ["MPI_Allgather()", "ConToPrim::Solve"]
    straggler_dict = filter_relevant_events(concat_df, straggler_events, 512)

    for e in straggler_events:
        arr = straggler_dict[e]
        arr = arr / 1e6
        arr = arr.astype(int).tolist()
        straggler_dict[e] = arr

    return straggler_dict


def run_plot_straggler_dict(run: str, run_dict: dict):
    time_sync = run_dict["MPI_Allgather()"]
    time_comp = run_dict["ConToPrim::Solve"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(time_sync, label="MPI_Allgather()", zorder=2)
    ax.plot(time_comp, label="ConToPrim::Solve", zorder=2)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (us)")
    ax.legend()
    ax.set_title(f"Straggler events in {run}")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.xaxis.grid(True, which="major", color="#bbb")
    ax.yaxis.grid(True, which="major", color="#bbb")
    ax.yaxis.grid(True, which="minor", color="#ddd")

    fig.tight_layout()

    plot_fname = f"straggler_events_{run}.png"
    PlotSaver.save(fig, "", None, plot_fname)


def run_lesson1():
    straggler_run = "profile10"
    straggler_dict = run_get_straggler_dict(straggler_run)
    PlotSaver.save_json(straggler_dict, "", None, f"straggler_dict_{straggler_run}")
    # run_plot_straggler_dict(straggler_run, straggler_dict)


def get_event_dict(run: str) -> dict:
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    trace_dir = trace_dir_fmt.format(run)

    run_df = read_all_pprof_simple(trace_dir)
    top_events = get_top_events(run_df, cutoff=0.01)

    event_dict = {}
    for e in top_events:
        event_dict[e] = get_event_array(run_df, e, 512)

    return event_dict


def simplify_event_dict(event_dict: dict) -> dict:
    new_dict = {}
    for k, v in event_dict.items():
        vtime = (v / 1e6).astype(int)
        k_class = classify_key(k)
        if k_class in new_dict:
            new_dict[k_class] += vtime
        else:
            new_dict[k_class] = vtime

    return new_dict


def run_lesson2():
    run = "athenapk13"
    run_dict = get_event_dict(run)
    run_dict_simple = simplify_event_dict(run_dict)

    for k, v in run_dict.items():
        run_dict[k] = (v/2).tolist()

    for k, v in run_dict_simple.items():
        run_dict_simple[k] = (v/2).tolist()

    PlotSaver.save_json(run_dict, "", None, f"event_dict_1000_{run}")
    PlotSaver.save_json(run_dict_simple, "", None, f"event_dict_simple_1000_{run}")


def run():
    run_lesson2()
    pass


if __name__ == "__main__":
    plot_init()
    run()
