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


def read_rank_df(rank: int) -> pd.DataFrame:
    trace_dir = "/mnt/ltio/parthenon-topo/athenapk12"
    df_path = f"{trace_dir}/profile/perf.parsed.{rank}"

    cols = [ "time_us", "parth", "psm_1", "psm_2" ]
    df = pd.read_csv(df_path, names=cols, on_bad_lines="skip")

    df_agg = df.groupby(["parth", "psm_1", "psm_2"], as_index=False).agg(
        {"time_us": ["sum", "count"]}
    )

    df_agg["rank"] = rank

    cols_new = ["rank", "parth", "psm_1", "psm_2", "time_us"]
    cols_flat = cols_new[:-1] + ["time_us_sum", "time_us_count"]
    df_agg = df_agg[cols_new]
    df_agg.columns = cols_flat

    return df_agg


def plot_getslot_total(aggr_df):
    rank_df = aggr_df.groupby(["rank"], as_index=False).agg(
        {"time_us_sum": "sum", "time_us_count": "sum"}
    )

    fig, ax = plt.subplots(1, 1)
    dx = rank_df["rank"]
    dy = rank_df["time_us_sum"]
    dy2 = rank_df["time_us_count"]

    ax.plot(dx, dy, label="Time", color="C0", zorder=4)
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Getslot() Time By Rank")

    ax.set_ylim(bottom=0)
    # 1e5 = 1e3 for us to ms, 100 for 100 timesteps
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} ms".format(x / 1e5))
    )
    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax2 = ax.twinx()
    ax2.plot(dx, dy2, color="C1", label="Count", zorder=2)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f}K".format(x / 1e5))
    )
    ax2.set_ylabel("Invoke Count")

    fig.tight_layout()
    fig.legend()

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.set_frame_on(False)

    plot_fname = "getslot_total"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def plot_getslot_by_psm():
    aggr_df["psm_1"].unique()
    aggr_df["psm_2"].unique()

    tmp_df = aggr_df.groupby([ "psm_1", "psm_2" ], as_index=False).agg(
        {"time_us_sum": "sum", "time_us_count": "sum"}
    ).sort_values("time_us_count")

    func_key = {
        "am_ctl_getslot_long": "getslot_long",
        "am_ctl_getslot_med": "getslot_med",
        "am_ctl_getslot_pkt": "getslot_pkt",
        "psmi_amsh_long_reply": "amsh_long_reply",
        "psmi_amsh_short_request": "amsh_short_req",
        "am_send_pkt_short": "send_pkt_short",
        "ips_spio_transfer_frame": "spio_xfer_frame",
        "ips_proto_flow_flush_pio": "ips_proto_xxx",
        "ips_proto_send_ctrl_message": "ips_proto_xxx",
        "ips_proto_timer_ctrlq_callback": "ips_proto_xxx",
    }

    aggr_df["psm_1"] = aggr_df["psm_1"].map(lambda x: func_key[x])
    aggr_df["psm_2"] = aggr_df["psm_2"].map(lambda x: func_key[x])
    aggr_df["psm"] = aggr_df["psm_2"] + "->" + aggr_df["psm_1"]
    aggr_df
    psm_df = aggr_df.groupby([ "psm", "rank" ], as_index=False).agg(
        {"time_us_sum": "sum", "time_us_count": "sum"}
    )

    psm_pvtdf = psm_df.pivot(index='psm', columns='rank', values='time_us_sum')
    psm_keys = psm_pvtdf.index.to_list()
    psm_keys
    psm_mat = psm_pvtdf.to_numpy()
    psm_mat
    psm_mat = psm_mat / 1e2 # per_ts
    psm_mat = psm_mat.astype(int)
    psm_mat

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.clear()
    # normalize by timestep
    ax.stackplot(np.arange(0, 512), psm_mat / 1000, labels=psm_keys, zorder=2)
    ax.set_title("spin_lock time by psm-caller")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim([0, 30])

    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} ms".format(x))
    )

    ax.legend()

    fig.tight_layout()

    plot_fname = "getslot.stackbypsm"
    PlotSaver.save(fig, "", None, plot_fname)
    pass

def simplify_parth_stack(callchain: str):
    distinct_keys = [
        "IOWrapper",
        "LoadFromFile",
        "LoadBalancingAndAdaptiveMeshRefinement",
        "ReceiveBoundaryBuffers",
        "SendBoundBufs",
        "ReceiveBoundBufs",
        "StartReceiveBoundBufs",
        "ReceiveFluxCorrections",
        "LoadAndSendFluxCorrections",
        "CommBuffer::Send",
        "CommBuffer::TryStartReceive",
        "CommBuffer::IsAvailable"
    ]

    for key in distinct_keys:
        if key in callchain:
            return key
            pass

    return "Uncategorized"

def map_parth_stack(df, col):
    parth_stack_map = {
        "IOWrapper": "IO",
        "LoadFromFile": "IO",
        "LoadBalancingAndAdaptiveMeshRefinement": "LB",
        "ReceiveBoundaryBuffers": "Recv_BB",
        "SendBoundBufs": "Send_BB",
        "ReceiveBoundBufs": "Recv_BB",
        "StartReceiveBoundBufs": "Recv_BB",
        "ReceiveFluxCorrections": "Recv_FL",
        "LoadAndSendFluxCorrections": "Send_FL",
        "CommBuffer::Send": "Comm",
        "CommBuffer::TryStartReceive": "Comm",
        "CommBuffer::IsAvailable": "Comm",
        "Uncategorized": "Uncategorized"
    }

    df[col] = df[col].apply(simplify_parth_stack)
    df[col] = df[col].apply(lambda x: parth_stack_map[x])
    return df

def simplify_list(l):
    print(len(l))
    l = list(set(map(simplify_parth_stack, l)))
    print(len(l))
    for k in l[:20]:
        print(k)

    return l


def plot_getslot_by_parth():
    parth_df = map_parth_stack(aggr_df, "parth")
    parth_df = parth_df.groupby([ "parth", "rank" ], as_index=False).agg(
        {"time_us_sum": "sum", "time_us_count": "sum"}
    )
    parth_pvtdf = parth_df.pivot(index='parth', columns='rank', values='time_us_sum')
    parth_pvtdf = parth_pvtdf.fillna(0)

    parth_keys = list(parth_pvtdf.index)
    parth_keys
    parth_mat = parth_pvtdf.to_numpy()
    parth_mat
    parth_mat = parth_mat.astype(int)
    parth_mat
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.clear()
    # normalize by timestep
    ax.stackplot(np.arange(0, 512), parth_mat / 1e5, labels=parth_keys, zorder=2)
    ax.set_title("spin_lock time by parthenon-caller")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim([0, 30])

    ax.yaxis.grid(visible=True, which="major", color="#bbb")
    ax.yaxis.grid(visible=True, which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} ms".format(x))
    )

    ax.legend(ncol=4)

    fig.tight_layout()

    plot_fname = "getslot.stackbyparth"
    PlotSaver.save(fig, "", None, plot_fname)


def aggr_dfs():
    all_ranks = np.arange(0, 512)
    all_ranks

    aggr_df = None
    with multiprocessing.Pool(16) as p:
        all_dfs = p.map(read_rank_df, all_ranks)
        aggr_df = pd.concat(all_dfs)

    aggr_df
    aggr_df_out = "/mnt/ltio/parthenon-topo/athenapk12/getslot_perf.csv"
    aggr_df.to_csv(aggr_df_out, index=False)

    plot_getslot_total(aggr_df)
    pass


if __name__ == "__main__":
    plot_init()
    run()
