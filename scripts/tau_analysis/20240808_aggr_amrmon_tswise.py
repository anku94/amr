import base64
import glob
import io
import os
import re
import requests
import string
import time


import multiprocessing
import numpy as np
import pandas as pd

import matplotlib.figure as pltfig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import plot_init_big, PlotSaver
from io import BytesIO
from trace_common import TraceUtils, SingleTrace, TraceSuite


class ProfileData:
    def __init__(self, trace_dir: str):
        self.trace_dir: str = trace_dir

        lb_dfpath = f"{trace_dir}/lb_aggr.feather"
        self.lb_data: pd.DataFrame = pd.read_feather(lb_dfpath).astype({"id": int})
        self.lb_data["tsid"] = pd.factorize(self.lb_data["timestep"])[0]
        self.lb_data.set_index("id", inplace=True)

        names_dfpath = f"{trace_dir}/lb_names.feather"
        lb_names: pd.DataFrame = pd.read_feather(names_dfpath)
        self.name_map: dict[str, int] = lb_names.set_index("name")["id"].to_dict()

    def lookup_name(self, name: str) -> int:
        if name in self.name_map:
            return self.name_map[name]
        return -1

    def lookup_name_substr(self, name: str) -> int:
        name_match = [k for k in self.name_map.keys() if name in k]

        if len(name_match) == 0:
            return -1

        if len(name_match) > 1:
            print(f"Multiple matches for {name}: {name_match}")

        return self.name_map[name_match[0]]

    def get_matrix(self, name: str, approx: bool = False) -> np.ndarray:
        name_id = self.lookup_name(name)
        if approx and name_id == -1:
            name_id = self.lookup_name_substr(name)

        if name_id == -1:
            print(f"Name {name} not found")
            return np.array([])

        df_name = self.lb_data.loc[name_id]
        mat_df = df_name.pivot(index="tsid", columns="rank", values="time")
        mat = mat_df.to_numpy()

        return mat


def send_to_server(fig: pltfig.Figure) -> None:
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    response = requests.post(
        "http://127.0.0.1:5000/update_plot",
        json={"plot_data": plot_data},
        proxies={"http": None, "https": None},
    )
    if response.status_code != 200:
        print(f"Failed: resp code {response.status_code}")


def save_df_with_retry(df: pd.DataFrame, fpath: str):
    for attempt in range(100):
        try:
            df.to_feather(fpath)
            break
        except Exception as e:
            print(f"Error saving {fpath}: {e}, attempt {attempt}")
            time.sleep(3)


def get_rank(fpath: str) -> int:
    fname = os.path.basename(fpath)
    mobj = re.match(r".*_(\d+)\.feather", fname)
    assert mobj is not None
    return int(mobj.groups()[0])


def read_all_names(trace_path: str) -> pd.DataFrame:
    names_files = glob.glob(f"{trace_path}/trace/names_*.feather")
    names_files = sorted(names_files, key=get_rank)

    with multiprocessing.Pool(16) as pool:
        names = pool.map(pd.read_feather, names_files)

    for f, df in zip(names_files, names):
        df["rank"] = get_rank(f)

    all_names = pd.concat(names)
    all_names["id"] = all_names.index
    all_names.rename(columns={"id": "id_old"}, inplace=True)

    unique_id = all_names[["name"]].drop_duplicates().reset_index(drop=True)
    unique_id["id_new"] = unique_id.index

    all_names = all_names.merge(unique_id, on="name", how="left")

    return all_names


def read_data_single(data_file: str) -> pd.DataFrame:
    print(f"Reading {data_file}")

    df = pd.read_feather(data_file)
    df = (
        df[["timestep", "id", "time"]]
        .groupby(["timestep", "id"])
        .agg({"time": "sum"})
        .reset_index()
    )
    df["rank"] = get_rank(data_file)

    return df


def read_data(trace_path: str):
    data_files = glob.glob(f"{trace_path}/trace/lb_aggr_*.feather")
    data_files = sorted(data_files, key=get_rank)

    # data_files = data_files[:1024]

    with multiprocessing.Pool(16) as pool:
        data = pool.map(read_data_single, data_files)

    all_data = pd.concat(data)

    return all_data


def gen_lb_aggr_wnames(trace_path: str):
    all_names = read_all_names(trace_path)
    all_data = read_data(trace_path)

    slim_names = all_names[["rank", "id_old", "id_new"]].copy()

    merged_data = all_data.merge(
        slim_names, left_on=["rank", "id"], right_on=["rank", "id_old"], how="left"
    )
    merged_data.drop(columns=["id", "id_old"], inplace=True)
    merged_data.rename(columns={"id_new": "id"}, inplace=True)

    lb_aggr = merged_data[["timestep", "id", "rank", "time"]].sort_values(
        ["timestep", "id", "rank"]
    )

    lb_aggr = (
        lb_aggr.groupby(["timestep", "id", "rank"]).agg({"time": "sum"}).reset_index()
    )

    lb_names = all_names[["name", "id_new"]].drop_duplicates().reset_index(drop=True)
    assert isinstance(lb_names, pd.DataFrame)
    lb_names.rename(columns={"id_new": "id"}, inplace=True)

    lb_aggr_out = f"{trace_path}/lb_aggr.feather"
    print(f"Writing to {lb_aggr_out}")
    lb_aggr.to_feather(lb_aggr_out)

    lb_names_out = f"{trace_path}/lb_names.feather"
    print(f"Writing to {lb_names_out}")
    lb_names.to_feather(lb_names_out)


def plot_mats(mat_key: str, mat_a: np.ndarray, mat_b: np.ndarray) -> tuple:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    lab_a = "h25"
    lab_b = "lpt"
    nts = mat_a.shape[0]

    axes[0].plot(np.mean(mat_a, axis=1), label=f"{lab_a} avg", color="C0")
    axes[0].plot(np.mean(mat_b, axis=1), label=f"{lab_b} avg", color="C1")
    mean_a = np.mean(mat_a)
    mean_b = np.mean(mat_b)
    axes[0].plot([-100, nts], [mean_a, mean_a], "--", color="C0")
    axes[0].plot([-100, nts], [mean_b, mean_b], "--", color="C1")

    axes[1].plot(np.max(mat_a, axis=1), label=f"{lab_a} max", color="C0")
    axes[1].plot(np.max(mat_b, axis=1), label=f"{lab_b} max", color="C1")
    mean_max_a = np.mean(np.max(mat_a, axis=1))
    mean_max_b = np.mean(np.max(mat_b, axis=1))
    axes[1].plot([-100, nts], [mean_max_a, mean_max_a], "--", color="C0")
    axes[1].plot([-100, nts], [mean_max_b, mean_max_b], "--", color="C1")

    axes[0].set_title(f"{mat_key} - {lab_a} vs {lab_b} - avg")
    # axes[1].set_title(f"{key} - {lab_a} vs {lab_b} - max")
    # turn off xlabel and xticks for axes[0]

    for ax in axes:
        ax.legend(ncol=2)
        ax.set_ylabel("Time (ms)")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f} ms")
        )
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=10))

        ax.grid(which="major", color="#bbb")
        ax.grid(which="minor", color="#ddd")
        ax.set_axisbelow(True)

    axes[0].tick_params(labelbottom=False)
    axes[0].set_title(f"{mat_key} - {lab_a} vs {lab_b}")
    axes[1].set_xlabel("Timestep (lbonly)")

    fig.tight_layout()

    return fig, axes

    plot_fname = f"{fname_prefix}_h25_lpt_bw24096.14"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_mat_imshow(mat: np.ndarray, key: str):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    im = ax.imshow(mat, aspect="auto")

    send_to_server(fig)

    fig.clear()
    ax = fig.add_subplot(111)
    pass


def run_analyze_aggr(tp_a: str, tp_b: str):
    pd_a = ProfileData(tp_a)
    pd_b = ProfileData(tp_b)

    all_keys = [
        # ("s7", "Step 7"),
        # ("s8", "Step 8"),
        # ("s9", "Step 9"),
        # ("mbar", "MPI_Barrier"),
        ("init", "Mesh::Initialize"),
    ]

    pref, k = all_keys[0]

    for pref, k in all_keys[-1:]:
        kmata = pd_a.get_matrix(k, approx=True)
        kmatb = pd_b.get_matrix(k, approx=True)

        # exclude first timestep, is messy
        kmata = kmata[1:, :]
        kmatb = kmatb[1:, :]

        xmax = kmata.shape[0]

        fig, ax = plot_mats(k, kmata, kmatb)
        ax[0].set_xlim([-10, xmax])
        ax[1].set_xlim([-10, xmax])
        plot_fname = f"{pref}_h25_lpt_bw4096.17"
        PlotSaver.save(fig, "", None, plot_fname)


def interactive_plotting(
    tp_a: str, tp_b: str, kmata: np.ndarray, kmatb: np.ndarray, key: str
):
    k = "Mesh::Initialize"
    kmata = pd_a.get_matrix(k, approx=True)
    kmatb = pd_b.get_matrix(k, approx=True)

    pd_a.lb_data[["timestep", "tsid"]].drop_duplicates().iloc[250:270]

    np.max(kmata, axis=1) / 1e3

    # ------------------------
    kmata = kmata[1:382, :]
    kmatb = kmatb[1:382, :]
    fig, axes = plot_mats(k, kmata, kmatb)
    axes[0].set_xlim(-10, 400)
    axes[1].set_xlim(-10, 400)
    PlotSaver.save(fig, "", None, "init_h25.16_vs_lpt")
    # ------------------------

    kmata.shape
    kmatb.shape

    kmata = kmata[350:450, :]
    kmatb = kmatb[350:450, :]

    np.max(kmata[377:380, :], axis=1)
    np.max(kmatb[370:380, :], axis=1)

    fig, axes = plot_mats(k, kmata, kmatb)
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x+350}"))
    axes[0].set_xlim(-10, 100)
    axes[1].set_xlim(-10, 100)
    PlotSaver.save(fig, "", None, "init_zoomed_h25_lpt_bw24096.14")

    # ------------------------

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.plot(kmata[376], label="lb# h25-376")
    ax.plot(kmata[377], label="lb# h25-377")
    ax.legend()

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f} ms"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.set_axisbelow(True)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Mesh::Initialize with h25 - lb #376/377")

    # ------------------------

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.plot(kmata[377], label="lb# h25-377")
    ax.plot(kmatb[377], label="lb# lpt-377")
    ax.legend()

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f} ms"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.set_axisbelow(True)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Mesh::Initialize - lb #377 - h25 vs lpt")

    PlotSaver.save(fig, "", None, "meshinit_h25_lpt_377")

    # ------------------------

    key = "Task_ReceiveBoundBufs"
    pd_a.name_map
    mat_rbb_a = pd_a.get_matrix(key, approx=True)
    mat_rbb_b = pd_b.get_matrix(key, approx=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.plot(mat_rbb_a[377], label="lb# h25-377")
    ax.plot(mat_rbb_b[377], label="lb# lpt-377")
    ax.legend()

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f} ms"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.set_axisbelow(True)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (ms)")
    ax.set_title("RecvBB - lb #377 - h25 vs lpt")

    send_to_server(fig)
    PlotSaver.save(fig, "", None, "rbb_h25_lpt_377")

    # ------------------------

    key = "Task_LoadAndSendBoundBufs"
    pd_a.name_map
    mat_bsbb_a = pd_a.get_matrix(key, approx=True)
    mat_bsbb_b = pd_b.get_matrix(key, approx=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.plot(mat_bsbb_a[377], label="lb# h25-377")
    ax.plot(mat_bsbb_b[377], label="lb# lpt-377")
    ax.legend()

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f} ms"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    ax.set_axisbelow(True)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Time (ms)")
    ax.set_title("BuildSendBB - lb #377 - h25 vs lpt")

    send_to_server(fig)
    PlotSaver.save(fig, "", None, "bsbb_h25_lpt_377")

    # ------------------------

    trace_path = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.16.perfdbg.hybrid25"
    pdt = ProfileData(trace_path)
    rbb_mat = pdt.get_matrix("Task_ReceiveBoundBufs")
    np.where(rbb_mat[380] > 75123)
    pdt.lb_data[["timestep", "tsid"]].drop_duplicates().iloc[350:400]

    np.max(rbb_mat, axis=1)
    np.max(rbb_mat[376])
    np.where(rbb_mat[376] > 68000)
    np.max(rbb_mat, axis=1)[376]
    rbb_mat[376][3482]
    list(np.where(rbb_mat > 75000)[0])
    np.where(rbb_mat[290] > 70000)
    rbb_mat.shape

    # get idxes where kmata377 > 50000
    ranks = np.where(kmata[377] > 50000)[0]
    ranks
    all_dfs = map(lambda x: pd.read_feather(f"{tp_a}/trace/lb_aggr_{x}.feather"), ranks)
    all_names = map(lambda x: pd.read_feather(f"{tp_a}/trace/names_{x}.feather"), ranks)

    all_dfs = list(all_dfs)
    all_names = list(all_names)

    # rank 512 is implicated
    rank = ranks[1]
    df = pd.read_feather(f"{tp_a}/trace/lb_aggr_{rank}.feather")
    names = pd.read_feather(f"{tp_a}/trace/names_{rank}.feather")

    tmp = all_dfs[0][["timestep", "tsid"]].drop_duplicates()
    tmp.iloc[350:400]

    for df in all_dfs:
        df["tsid"] = pd.factorize(df["timestep"])[0]

    for names, df in zip(all_names, all_dfs):
        tsdf = df[df["tsid"] == 377]
        tsdf = tsdf.merge(names, left_on="id", right_on="id")
        print(tsdf[tsdf["name"] == "Task_ReceiveBoundBufs"])

    tsdf = df[df["tsid"] == 377]
    tsdf = tsdf.merge(names, left_on="id", right_on="id")
    tsdf[tsdf["name"] == "Task_ReceiveBoundBufs"]
    # lb# 377 corresponds to timestep 3074
    print(tsdf.to_string())

    # --------------

    mat_rbb_a.shape
    np.max(mat_rbb_a, axis=1) / 1e3
    np.where(mat_rbb_a[12, :] > 50000)[0].shape


def run():
    # trace_key = "mpidbg4096ub22.13"
    # suite = TraceUtils.get_traces(trace_key)
    # t = suite.traces[0]
    # rank = 32

    trace_path = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.14.perfdbg.hybrid25"
    trace_path = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.15.perfdbg.hybrid25"
    trace_path = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.16.perfdbg.hybrid25"
    tp_a = trace_path
    trace_path = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.17.perfdbg.hybrid25"
    trace_path = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.17.perfdbg.hybrid25"
    trace_path_pref = "/mnt/ltio/parthenon-topo/ub22perfdbg"
    tp_a = f"{trace_path_pref}/blastw4096.18.perfdbg.hybrid25"
    tp_b = f"{trace_path_pref}/blastw4096.18.perfdbg.lpt"
    # parse_rank(trace_path, 32)

    # First, parse via run_mpich using one rank per file
    # Then aggregate using this:
    gen_lb_aggr_wnames(tp_a)

    # run_analyze_aggr(tp_a, tp_b)


if __name__ == "__main__":
    # plot_init_big()
    run()
