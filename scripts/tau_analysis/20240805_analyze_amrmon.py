import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import os
from common import PlotSaver

def read_amrmon(run_dir: str) -> dict[str, np.ndarray]:
    amrmon_file = f"{run_dir}/trace/amrmon_rankwise.txt"
    amrmon_data = open(amrmon_file, "r").readlines()
    amrmon_data = [x.strip() for x in amrmon_data]

    phases = amrmon_data[::2]
    phase_data = amrmon_data[1::2]
    str2arr = lambda x: np.array([float(x) for x in x.split(",")])
    all_phase_data: dict[str, np.ndarray] = {
        k: str2arr(v) for k, v in zip(phases, phase_data)
    }

    return all_phase_data


def get_max_outliers(data: np.ndarray) -> np.ndarray:
    p99 = np.percentile(data, 99)
    return np.where(data > p99)[0]


def get_min_outliers(data: np.ndarray) -> np.ndarray:
    p1 = np.percentile(data, 1)
    return np.where(data < p1)[0]


def plot_mesh_init():
    fig, ax = plt.subplots(1, 1, figsize=(32, 8))

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#eee")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    return fig, ax


def run():
    perf_run = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.18.perfdbg"
    perf_dir_cdp = f"{perf_run}.cdpc512par8"
    perf_dir_h25 = f"{perf_run}.hybrid25"

    run_dir = perf_dir_cdp

    nranks = 4096
    nchunks = 16

    data_cdp = read_amrmon(perf_dir_cdp)
    data_h25 = read_amrmon(perf_dir_h25)

    outlier_func = "Mesh::Initialize"
    ocdp = get_max_outliers(data_cdp[outlier_func])
    oh25 = get_max_outliers(data_h25[outlier_func])
    ocdp
    oh25
    ocommon = list(set(ocdp) & set(oh25))
    ocommon
    ocommon = [3576, 3579]
    compute_rank_ptile = lambda arr, x: np.sum(arr < arr[x]) / len(arr)

    data_cdp.keys()

    all_funcs = [
        "Mesh::Initialize",
        "RedistributeAndRefineMeshBlocks",
        "Task_ReceiveFluxCorrections",
        "MPI_Barrier",
        "Task_LoadAndSendBoundBufs",
        "Task_BuildSendBoundBufs",
        "MPI_Allgather",
        "MPI_Allgatherv",
    ]

    data_cdp["Mesh::Initialize"]

    for func in all_funcs:
        print(f"Function: {func}")
        for r in ocommon:
            cdp_arr = data_cdp[func]
            h25_arr = data_h25[func]

            cdp_ptile = compute_rank_ptile(cdp_arr, r)
            h25_ptile = compute_rank_ptile(h25_arr, r)

            print(f" - Rank {r:4d}: {cdp_ptile:2.2%} (cdp) vs {h25_ptile:2.2%} (h25)")

    all_funcs = [
            ("Mesh::Initialize", "meshinit"),
            ("Task_BuildSendBoundBufs", "bsend_bb"),
            ("Task_LoadAndSendBoundBufs", "lsend_bb"),
            ("Task_LoadAndSendBoundBufs_Rebuild", "lsend_bb_reb"),
            ("Task_LoadAndSendBoundBufs_Restrict", "lsend_bb_restr"),
            ("Task_LoadAndSendBoundBufs_Load", "lsend_bb_load"),
            ("Task_LoadAndSendBoundBufs_SendNoFence", "lsend_bb_sendnf"),
            ("Task_ReceiveFluxCorrections", "recv_fc"),
            ("Task_ReceiveBoundBufs", "recv_bb"),
            ]

    fig, ax = plot_mesh_init()
    # func, func_name = all_funcs[2]
    # mi_cdp = data_cdp[func]
    # mi_h25 = data_h25[func]
    # diff = (mi_h25 - mi_cdp)/1e6
    # diff[200]
    # func
    #
    # ax.plot(mi_cdp, label="CDP")
    # ax.plot(mi_h25, label="H25")
    # ax.plot(mi_h25 - mi_cdp, label="H25 - CDP")

    def add_diff(func_idx: int):
        func, func_name = all_funcs[func_idx]
        func_diff = (data_h25[func] - data_cdp[func])
        ax.plot(func_diff, label=f"{func_name} diff")

    for fidx in [2, 6]:
        add_diff(fidx)

    plot_fname = os.path.basename(perf_run)
    # plot_fname = f"{plot_fname}.{func_name}.png"
    plot_fname = f"{plot_fname}.psmdbg.las_breakdown"
    # ax.set_title(f"{func} - {plot_fname}")
    ax.set_title(plot_fname)
    ax.legend()
    PlotSaver.save(fig, "", None, plot_fname)
    


if __name__ == "__main__":
    run()
