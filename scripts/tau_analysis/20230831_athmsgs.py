import matplotlib
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import re

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analyze_msgs import aggr_msgs_all, join_msgs, get_relevant_pprof_data
from common import plot_init_big as plot_init, PlotSaver
from typing import Tuple, List


def set_interactive():
    matplotlib.use("GTK3Agg")
    plt.ion()


def setup_ax_default(ax):
    ax.xaxis.grid(visible=True, which="major", color="#999")
    ax.xaxis.grid(visible=True, which="minor", color="#ddd")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(32))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(8))

    ax.yaxis.grid(visible=True, which="major", color="#aaa")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_ylim(bottom=0)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    )
    pass


def get_msg_dfs(trace_name: str) -> Tuple:
    trace_path = f"/mnt/ltio/parthenon-topo/{trace_name}/trace"
    chan_df_path = f"{trace_path}/msgs.aggr.chan.csv"
    send_df_clipped_path = f"{trace_path}/msgs.aggr.send.clipped.csv"
    send_df_path = f"{trace_path}/msgs.aggr.send.csv"

    if os.path.exists(send_df_clipped_path):
        print("Clipped send_df found - using that.")
        send_df_path = send_df_clipped_path
    else:
        print("ALERT! send_df_clipped does not exist. Reading the full thing.")

    if os.path.exists(send_df_path) and os.path.exists(chan_df_path):
        print("Reading chan_df and send_df from aggr files")
        chan_df = pd.read_csv(chan_df_path)
        send_df = pd.read_csv(send_df_path)
    else:
        print("ALERT! Aggr files for send/chan don't exist. Reading bins")
        chan_df, send_df = aggr_msgs_all(trace_name)

    max_ts = send_df["ts"].max()
    max_rank = send_df["rank"].max()
    print(f"Send_df. Max_ts: {max_ts}, Max_rank: {max_rank}")

    return chan_df, send_df


def get_mats(trace_name: str, ts: int):
    chan_df, send_df = get_msg_dfs(trace_name)
    send_df = send_df[send_df["ts"] == ts]
    msg_df = join_msgs(send_df, chan_df)

    blk_mat_df = msg_df.groupby(["blk_id", "nbr_id"], as_index=False).agg(
        {"msgsz": "count"}
    )
    blk_mat = (
        blk_mat_df.pivot(index="blk_id", columns="nbr_id", values="msgsz")
        .fillna(0)
        .astype(int)
    )

    rank_mat_df = msg_df.groupby(["blk_rank", "nbr_rank"], as_index=False).agg(
        {"msgsz": "count"}
    )

    rank_mat = (
        rank_mat_df.pivot(index="blk_rank", columns="nbr_rank", values="msgsz")
        .fillna(0)
        .astype(int)
    )

    sym_test = sum(blk_mat.sum(axis=1) - blk_mat.sum(axis=0))
    if sym_test != 0:
        print("message exchange is not symmetric!!")

    return blk_mat, rank_mat


def max_non_zero_column_per_row(matrix):
    non_zero_mask = matrix != 0
    max_columns = np.max(
        np.where(non_zero_mask, np.arange(matrix.shape[1]), -1), axis=1
    )
    return max_columns


def plot_1():
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    dyt = kfls + kflr + kbcs + kbcr
    dx = range(len(dyt))

    ax.plot(dx, dyt)
    setup_ax_default(ax)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Rank ID")
    ax.set_title("Msg Time vs Comm Time (FL_S + FL_R + BC_S + BC_R)")

    fig.tight_layout()
    plot_fname = f"{trace_name}_flbc_sndrcv"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def plot_2():
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    dyt = kfls + kbcs
    dx = range(len(dyt))

    ax.plot(dx, dyt)
    setup_ax_default(ax)

    ax.set_ylabel("Time (s) (blue)")
    ax.set_xlabel("Rank ID")
    ax.set_title("Msg Time vs Comm Time (Send)")

    ax2 = ax.twinx()
    ax2.plot(dx, dyn, color="red")
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Msgs Sent (red)")
    ax2.set_ylim(bottom=0)

    # to align ax1 and ax2
    ax.set_ylim([35 * 1e6, 115 * 1e6])

    fig.tight_layout()
    plot_fname = f"{trace_name}_flbc_snd.vs.msgcnt"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def plot_msg_time_corr():
    trace_name = "athenapk13"
    pprof_data = get_relevant_pprof_data(trace_name)

    getslot_df = pd.read_csv(f"/mnt/ltio/parthenon-topo/{trace_name}/getslot.csv")
    getslot_df
    dygs = getslot_df["usec_sum"]

    blk_mat, rank_mat = get_mats(trace_name, 4)

    rank_msgs = rank_mat.sum(axis=0).to_numpy()
    rank_msgs.size

    kbcs = pprof_data["kbcs"]
    kbcr = pprof_data["kbcr"]
    kfls = pprof_data["kfls"]
    kflr = pprof_data["kflr"]

    plot_1()
    plot_2()

    send_time = pprof_data["kbcs"]
    recv_time = pprof_data["kbcr"]
    mip_time = pprof_data["mip"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_xlim([192, 512])
    ax.set_xlim([0, 512])
    ax.clear()
    ax2.clear()

    dyt = send_time
    dyt = mip_time
    dyt = pprof_data["kbcs"] + pprof_data["kfls"]
    dyt = kfls + kflr
    dyt = kfls + kflr + kbcs + kbcr
    dyn = rank_msgs
    dx = range(len(dyt))

    ax.plot(dx, dyt)
    ax2 = ax.twinx()
    ax2.plot(dx, dyn, color="red")
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Time (us)")
    ax.set_ylabel("Msgs Sent")

    setup_ax_default(ax)

    ax.set_ylim([35 * 1e6, 115 * 1e6])
    ax2.set_ylim(bottom=0)

    ax.set_title("Msg Exchange Count vs Comm Time (Send)")

    fig.tight_layout()

    pass


def idk():
    trace_name = "athenapk10"
    pprof_data = get_relevant_pprof_data(trace_name)
    blk_mat, rank_mat = get_mats("athenapk6", 3)
    plt.ion()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(blk_mat)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(rank_mat)

    max_nz_blk = max_non_zero_column_per_row(blk_mat)
    max_nz_rank = max_non_zero_column_per_row(rank_mat)

    dx = np.arange(512)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(dx, max_nz_rank)

    ax2 = ax.twinx()
    ax2.clear()
    dy = pprof_data["kbcs"] + pprof_data["kbcr"]
    dy = pprof_data["kbcs"]
    ax2.plot(dx, dy, color="red")

    msg_dy = rank_mat.sum(axis=1)
    ax.clear()
    ax.plot(dx, msg_dy)
    ax.set_ylim([0, 600])
    ax2.set_ylim([0, 6e8])

    ax.xaxis.grid(visible=True, which="major", color="#bbb")
    ax.xaxis.grid(visible=True, which="minor", color="#ddd")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(32))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(8))

    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    )


def run():
    idk()
    pass


if __name__ == "__main__":
    plot_init()
    run()
