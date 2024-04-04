import os
import glob
import pandas as pd
import re
import sys

import multiprocessing
import numpy as np

from common import PlotSaver, plot_init_big
import matplotlib.pyplot as plt


def read_pprof_line(line: str) -> list[str]:
    mobj = re.match(r'\"(.*)\"(.*)', line)
    if mobj == None:
        return []
    mobj.groups()
    key, props = mobj.groups()
    key = key.strip()
    props = props.strip().split(' ')
    return [key] + props


def read_map(map_path: str) -> dict[int, int]:
    with open(map_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(",") for line in lines]
        lines = [(int(line[0]), int(line[1])) for line in lines]
        return dict(lines)


def read_profile(prof_file: str):
    # prof_dir = "/users/ankushj/repos/amr/scripts/ember-run"
    # prof_file = f"{prof_dir}/profile.0.0.0"
    fdata = open(prof_file).readlines()
    lines = [line.strip() for line in fdata]

    # find index of line with "cumulative" in import
    idx = [i for i, s in enumerate(lines) if s.startswith("# eventname")][0]
    header = lines[idx].strip('# ').split()
    data = list(map(read_pprof_line, lines[idx + 1:]))

    df = pd.DataFrame.from_records(data, columns=header)
    # prof_file is of the form profile.<rank>.<thread>.<node>
    # get rank
    df["rank"] = prof_file.split('.')[-3]
    return df


def read_all_profiles(prof_dir: str):
    print(f"Reading profiles from {prof_dir}...")
    prof_files = glob.glob(f"{prof_dir}/profile.*")
    print(f"Found {len(prof_files)} profile files.")

    nthreads = 16
    with multiprocessing.Pool(nthreads) as pool:
        df_list = pool.map(read_profile, prof_files)
        df = pd.concat(df_list)
    # df = pd.concat([read_profile(prof_file) for prof_file in prof_files])
    return df


def construct_send_recv_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df_match = df[df["eventname"].str.contains("Message size sent to node")]
    df_match = df_match[~df_match["eventname"].str.contains(" : ")]
    df_match["dest"] = df_match["eventname"].str.split(" ").str[-1]

    print(f"\ndf_match: \n{df_match}")

    if len(df_match) == 0:
        raise Exception("No message data found in profile.")
#
    # convert cols dest, mean, numevents to int
    df_match = df_match.astype(
        {"rank": int, "dest": int, "mean": float, "numevents": int})

    df_match["comm_size"] = df_match["mean"] * df_match["numevents"]
    df_msg = df_match[['rank', 'dest', 'comm_size']].copy()

    return df_msg


def construct_matrix_from_pairs(df: pd.DataFrame) -> np.ndarray:
    df = df.groupby(['rank', 'dest']).sum().reset_index()
    df = df.pivot(index='rank', columns='dest', values='comm_size').fillna(0)
    df_mat = df.astype(float).to_numpy()
    print(df_mat)
    return df_mat


def plot_matrix(df_mat: np.ndarray, plot_fname: str, plot_label: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(df_mat, cmap='plasma', aspect='auto')
    ax.set_xlabel("Destination Rank")
    ax.set_ylabel("Source Rank")
    ax.set_title(f"Comm. Matrix for {plot_label}")
    plot_fname = f"comm_matrix_{plot_fname}"
    PlotSaver.save(fig, "", None, plot_fname)


def compute_local_comm(msg_df: pd.DataFrame) -> tuple[float, float]:
    ranks_per_node = 16
    def node(x): return int(x / ranks_per_node)
    local_comm = 0.0
    all_comm = 0.0

    for _, row in msg_df.iterrows():
        src, dest, msg_size = row
        src_node = node(src)
        dest_node = node(dest)

        all_comm += msg_size
        if src_node == dest_node:
            local_comm += msg_size

    return (local_comm, all_comm)


def compute_local_comm_fast(msg_df: pd.DataFrame) -> tuple[float, float]:
    ranks_per_node = 16
    def node(x): return int(x / ranks_per_node)
    msg_df["src_node"] = msg_df["rank"].apply(node)
    msg_df["dest_node"] = msg_df["dest"].apply(node)
    all_comm = msg_df["comm_size"].sum()
    local_comm = (msg_df[msg_df["src_node"] == msg_df["dest_node"]])[
        "comm_size"].sum()

    return (local_comm, all_comm)


def plot_prof_dir(prof_dir: str, prof_dirtype: str):
    prof_dir
    msg_mat
    msg_mat_2
    msg_mat_2 - msg_mat * 400
    comm_df = read_all_profiles(prof_dir)
    msg_df = construct_send_recv_pairs(comm_df)
    msg_mat = construct_matrix_from_pairs(msg_df)

    prof_dirname = os.path.basename(prof_dir)
    # prof_dirtype = "halo3d"
    # if "ember-runs-26" in prof_dirname:
    #     prof_dirtype = "halo3d_26"
    plot_fname = f"comm_matrix_{prof_dirtype}_{prof_dirname}"
    plot_matrix(msg_mat, plot_fname)


def plot_comm_tuples(traces: list[str], comm_tuples: dict[str, tuple[float, float]], trace_labels: dict[str, str]):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    labels_x = [trace_labels[t] for t in traces]
    data_x = np.arange(len(traces))
    width = 0.35

    data_yloc = np.array([t[0] for t in comm_tuples.values()])
    data_ytot = np.array([t[1] for t in comm_tuples.values()])

    data_yloc = data_yloc / (2 ** 40)  # to terabytes
    data_ytot = data_ytot / (2 ** 40)  # to terabytes

    ax.bar(data_x, data_yloc, width, label="Local Comm", color="C0", zorder=2)
    ax.bar(data_x + width, data_ytot, width, label="All Comm", color="C1", zorder=2)
    ax.set_xticks(data_x + width / 2)
    ax.set_xticklabels(labels_x)

    #ax.yaxis.grid(True)
    ax.yaxis.grid(True, which='both')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, loc: "{:.1f}TB".format(x)))

    ax.set_xlabel("Trace")
    ax.set_ylabel("Comm. Volume")
    ax.set_title("Comm. Volume for Different Traces")
    ax.legend()
    fig.tight_layout()
    PlotSaver.save(fig, "", None, "comm_volume")


def run_plot():
    prof_dir = "/mnt/ltio/ember-runs/rval510.run0.iter100.var1"
    plot_prof_dir(prof_dir, "halo3d")

    prof_dir = "/mnt/ltio/ember-runs/rval384.run0.iter100.var1"
    plot_prof_dir(prof_dir, "halo3d")

    prof_dir = "/mnt/ltio/ember-runs/rval256.run0.iter100.var1"
    plot_prof_dir(prof_dir, "halo3d")

    prof_dir = "/mnt/ltio/ember-runs/rval128.run0.iter100.var1"
    plot_prof_dir(prof_dir, "halo3d")

    prof_dir = "/mnt/ltio/ember-runs/rval0.run0.iter100.var1"
    plot_prof_dir(prof_dir, "halo3d")


def run_plot_v2():
    prof_dir = "/mnt/ltio/ember-runs-v2/rval0.run0.iter100.var4"
    plot_prof_dir(prof_dir, "halo3dv2")

    prof_dir = "/mnt/ltio/ember-runs-v2/rval510.run0.iter100.var4"
    plot_prof_dir(prof_dir, "halo3dv2")


def run_stochsg():
    trace_names = ["stochsg44", "stochsg45", "stochsg46", "stochsg47"]
    trace_labels = ["Baseline", "LPT", "ContigDP", "ContigDP+Iter"]
    trace_labels = dict(zip(trace_names, trace_labels))

    all_comm_tuples = {}

    trace_name = "stochsg44"

    for trace_name in trace_names:
        prof_dir = f"/mnt/ltio/parthenon-topo/{trace_name}/profile"
        comm_df = read_all_profiles(prof_dir)
        msg_df = construct_send_recv_pairs(comm_df)
        comm_mat = construct_matrix_from_pairs(msg_df)
        comm_mat_mb = (comm_mat / (2 ** 20)).astype(int)
        comm_mat_mb.shape
        comm_mat_mb
        comm_mat_mb.sum(axis=1)

        plot_matrix(comm_mat, trace_name, trace_labels[trace_name])
        comm_tuple = compute_local_comm_fast(msg_df)
        all_comm_tuples[trace_name] = comm_tuple
        del comm_df, msg_df, comm_mat

    all_comm_tuples
    plot_comm_tuples(trace_names, all_comm_tuples, trace_labels)


def run(prof_dir: str):
    # prof_dir = "/users/ankushj/repos/amr/scripts/ember-run"

    # print(f"Local comm: {local_comm}, All comm: {all_comm}")
    # print(f"Local comm ratio: {local_comm/all_comm}")

    print("{:.0f},{:.0f},{:.2f}".format(
        local_comm, all_comm, local_comm/all_comm))
    # msg_df
    # len(msg_df["rank"].unique())
    # len(msg_df["dest"].unique())

    # map_path = "/users/ankushj/repos/amr/scripts/ember_maps/map_h32_c16_r0.txt"
    # rank_map = read_map(map_path)
    # print(comm_mat)
    # print(rank_map)
    # pass


if __name__ == "__main__":
    plot_init_big()
    prof_dir = sys.argv[1]
    run(prof_dir)
    # run_plot_v2()
