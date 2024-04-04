import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import re
import subprocess
import struct
import sys
import time

#  from memory_profiler import profile

import ray
import traceback

from matplotlib.ticker import FuncFormatter, MultipleLocator
from pandas import DataFrame

from sklearn import linear_model
from typing import Dict, Tuple

from trace_reader import TraceOps

from task import Task

from analyze_pprof import (
    setup_plot_stacked_generic,
    read_all_pprof_simple,
    filter_relevant_events,
)


#  ray.init(address="h0:6379")
trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"


def setup_interactive():
    matplotlib.use("abc")
    matplotlib.use("TkAgg")
    matplotlib.use("WebAgg")
    matplotlib.use("GTK3Agg")
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    dx = np.arange(5)
    dy = np.arange(5, 0, -1)
    ax.plot(dx, dy)


class MsgAggrTask(Task):
    nworkers = 4
    cached = True

    def __init__(self, trace_dir):
        super().__init__(trace_dir)

    def gen_worker_fn_args(self):
        self.mgr = multiprocessing.Manager()
        self.ns = self.mgr.Namespace()

        phase_df, max_ts = self.get_phase_df(self.cached)
        self.ns.trace_dir = self._trace_dir
        self.ns.phase_df = phase_df
        self.ns.max_ts = max_ts

        fn_args = {"ns": self.ns}
        return fn_args

    @staticmethod
    def worker(fn_args):
        rank = fn_args["rank"]
        ns = fn_args["ns"]

        def log(msg):
            print("Rank {}: {}".format(rank, msg))

        return aggr_msgs_map(rank, ns.phase_df, ns.trace_dir, ns.max_ts)

    def get_phase_df(self, cached):
        phase_map = gen_phase_map(self._trace_dir, cached=cached)
        max_ts = min(map(lambda x: x.shape[0], phase_map.values()))

        t0 = time.time()
        phase_df = gen_phase_df(phase_map)
        #  phase_df = None
        t1 = time.time()

        self.log("Phase DF construction: {:.1f}s".format(t1 - t0))
        return phase_df, max_ts


def joinstr(x):
    return ",".join([str(i) for i in x])


def aggr_msgs(fn_args):
    trace_dir = fn_args["trace_dir"]
    rank = fn_args["rank"]

    rank_msgcsv = "{}/trace/msgs.rcv.{}.csv".format(trace_dir, rank)
    print(f"Reading {rank_msgcsv}")

    df = pd.read_csv(rank_msgcsv, sep="|")
    #  print(df)
    #  print(df["phase"].unique())

    df = df.groupby(["rank", "timestep", "phase", "send_or_recv"], as_index=False).agg(
        {"msg_sz": ["mean", "sum", "count"], "peer": ["nunique", joinstr]}
    )

    df.columns = ["_".join(col).strip("_") for col in df.columns.values]

    df.to_csv("{}/aggr/msgs.{}.csv".format(trace_dir, rank), index=None)
    df = None


def read_msgs(fpath):
    rank = int(re.search(r"msgs.(\d+).bin$", fpath).groups(0)[0])
    msgbin_data = open(fpath, "rb").read()

    print(f"Read messages: {rank}: {fpath}")

    # ptr, blk_id, blk_rank, nbr_id, nbr_rank, tag, is_flx
    chan_sz = 29
    chan_fmt = "@Piiiiic"
    chan_fmtc = struct.Struct(chan_fmt)
    assert chan_sz == struct.calcsize(chan_fmt)

    # tag, dest, sz, ts
    # ptr, bufsz, recv_rank, tag, timestamp
    send_sz = 28
    # can't use P with =, can't use @ because padding issues
    send_fmt = "=QiiiQ"
    send_fmtc = struct.Struct(send_fmt)
    assert send_sz == struct.calcsize(send_fmt)

    all_ts_data = []

    ptr = 0
    while ptr < len(msgbin_data):
        (ts,) = struct.unpack("@i", msgbin_data[ptr : ptr + 4])
        ptr += 4

        (chanbuf_sz,) = struct.unpack("@i", msgbin_data[ptr : ptr + 4])
        ptr += 4

        chan_recs = list(chan_fmtc.iter_unpack(msgbin_data[ptr : ptr + chanbuf_sz]))
        ptr += chanbuf_sz

        (sendbuf_sz,) = struct.unpack("@i", msgbin_data[ptr : ptr + 4])
        ptr += 4

        send_recs = list(send_fmtc.iter_unpack(msgbin_data[ptr : ptr + sendbuf_sz]))
        ptr += sendbuf_sz

        all_ts_data.append((ts, chan_recs, send_recs))

    chan_cols = ["ptr", "blk_id", "blk_rank", "nbr_id", "nbr_rank", "tag", "isflx"]
    send_cols = ["ptr", "msgsz", "Dest", "tag", "timestamp"]

    all_chan_df = []
    all_send_df = []

    for tup in all_ts_data:
        ts, chan_recs, send_recs = tup
        chan_df = pd.DataFrame.from_records(chan_recs, columns=chan_cols)
        chan_df["ts"] = ts

        send_df = pd.DataFrame.from_records(send_recs, columns=send_cols)
        send_df["ts"] = ts

        all_chan_df.append(chan_df)
        all_send_df.append(send_df)

    chan_cdf = pd.concat(all_chan_df)
    chan_cdf["isflx"] = chan_cdf["isflx"].apply(lambda x: int.from_bytes(x, "little"))

    send_cdf = pd.concat(all_send_df)

    chan_cdf["rank"] = rank
    send_cdf["rank"] = rank

    cols = chan_cdf.columns
    cols = ["rank", "ts"] + list(cols[:-2])
    chan_cdf = chan_cdf[cols]

    cols = send_cdf.columns
    cols = ["rank", "ts"] + list(cols[:-2])
    send_cdf = send_cdf[cols]

    return (chan_cdf, send_cdf)


def aggr_msgs_some(
    trace_name: str, rank_beg: int, rank_end: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_name)

    path_fmt = "{0}/trace/msgs/msgs.{1}.bin"
    all_bins = [path_fmt.format(trace_dir, r) for r in range(rank_beg, rank_end + 1)]
    print(f"Reading {all_bins} msgs.bin")

    for p in all_bins:
        assert os.path.exists(p)

    with multiprocessing.Pool(16) as p:
        all_dfs = p.map(read_msgs, all_bins)

    chan_df: DataFrame = pd.concat(map(lambda x: x[0], all_dfs))
    send_df = pd.concat(map(lambda x: x[1], all_dfs))

    chan_df.sort_values(["rank", "ts"], inplace=True)
    send_df.sort_values(["rank", "ts"], inplace=True)

    return chan_df, send_df


def aggr_msgs_all(trace_name):
    global trace_dir_fmt
    trace_dir = trace_dir_fmt.format(trace_name)

    print(f"Searching for msgs.*.bin in {trace_dir}")
    glob_patt = trace_dir + "/trace/msgs/msgs.*.bin"
    print(f"Glob path: {glob_patt}")
    all_bins = glob.glob(glob_patt)
    print(f"Bins found: {len(all_bins)}")

    #  all_bins = all_bins[:16]

    with multiprocessing.Pool(16) as p:
        all_dfs = p.map(read_msgs, all_bins)

    chan_df = pd.concat(map(lambda x: x[0], all_dfs))
    send_df = pd.concat(map(lambda x: x[1], all_dfs))

    chan_df = chan_df.iloc[:, [1, 0] + list(range(2, chan_df.shape[1]))]
    send_df = send_df.iloc[:, [1, 0] + list(range(2, send_df.shape[1]))]

    chan_df.sort_values(["ts", "rank"], inplace=True)
    send_df.sort_values(["ts", "rank"], inplace=True)

    chan_out = f"{trace_dir}/trace/msgs.aggr.chan.csv"
    print(f"Writing to {chan_out}")
    chan_df.to_csv(chan_out, index=None)

    send_out = f"{trace_dir}/trace/msgs.aggr.send.csv"
    print(f"Writing to {send_out}")
    send_df.to_csv(send_out, index=None)

    return chan_df, send_df


def join_msgs(send_df, chan_df):
    x = chan_df["ptr"].unique()
    y = send_df["ptr"].unique()
    # import pdb

    # pdb.set_trace()
    assert len(set(y).difference(set(x))) == 0

    all_chan_ts = chan_df["ts"].unique()
    all_chan_df = []

    max_ts = send_df["ts"].max()

    for ts in range(max_ts + 1):
        if ts in all_chan_ts:
            closest_ts = ts
        else:
            closest_ts = max([t for t in all_chan_ts if t < ts])

        print(f"For ts: {ts}, using ts {closest_ts}")
        df_ts = chan_df[chan_df["ts"] == closest_ts].copy()
        df_ts["ts"] = ts
        df_ts.drop_duplicates(subset=["rank", "ts", "ptr"], keep="last", inplace=True)
        all_chan_df.append(df_ts)

    chan_unroll_df = pd.concat(all_chan_df)
    joined_df = send_df.merge(chan_unroll_df, how="left", on=["ts", "ptr"])

    send_counts = send_df.groupby("ts", as_index=False).agg({"msgsz": "count"})
    print("-> Send counts: \n", send_counts)

    join_counts = joined_df.groupby("ts", as_index=False).agg({"msgsz": "count"})
    print("-> Join counts: \n", join_counts)
    return joined_df


def plot_imshow(mat, stat):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    vmin, vmax = np.percentile(mat, 1), np.percentile(mat, 99)
    bounds = np.linspace(vmin, vmax, 16)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")
    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")

    fig.subplots_adjust(left=0.15, right=0.78)
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])
    fig.colorbar(im, cax=cax)

    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Timestep")

    ax.set_title(f"Rank-Wise Heatmap (Stat: {stat})")
    plot_fname = f"msgs.aggr.rw.{stat.lower()}"

    trace_dir = ""
    #  PlotSaver.save(fig, trace_dir, None, plot_fname)


def plot_stat_slice(mat, stat, ts):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    data_y = mat[ts]
    data_x = range(len(data_y))

    ax.plot(data_x, data_y, zorder=2)
    ax.set_xlabel("Rank")
    ax.set_ylabel(f"Stat {stat}")
    ax.set_title(f"Stat {stat} vs Rank (TS: {ts})")

    ax.set_ylim(bottom=0)

    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    plot_fname = f"msgs.aggrslice.rw.{stat.lower()}"

    trace_dir = ""
    PlotSaver.save(fig, trace_dir, None, plot_fname)


def gen_phase_map(trace_dir, cached=False):
    pmap_pkl = "/users/ankushj/CRAP/.phase_map"
    if cached:
        with open(pmap_pkl, "rb") as f:
            phase_map = pickle.loads(f.read())
            return phase_map

    tr = TraceOps(trace_dir)
    ar1 = tr.multimat("tau:AR1")
    sr = tr.multimat("tau:SR")
    ar2 = tr.multimat("tau:AR2")
    ar3 = tr.multimat("tau:AR3-AR3_UMBT")
    #  ar1 = np.zeros([30572, 512])
    #  sr = np.zeros([30572, 512])
    #  ar2 = np.zeros([30572, 512])
    #  ar3 = np.zeros([30572, 512])

    phase_map = {
        "FluxExchange": ar1,
        "BoundaryComm": sr,
        "BoundaryApply": ar2,
        "LoadBalancing": ar3,
    }

    with open(pmap_pkl, "wb+") as f:
        f.write(pickle.dumps(phase_map))

    return phase_map


def sep_msgs(fn_args):
    trace_dir = fn_args["trace_dir"]
    rank = fn_args["rank"]

    def log(msg):
        print("Rank {}: {}".format(rank, msg))

    rank_msgcsv = "{}/trace/msgs.{}.csv".format(trace_dir, rank)
    rank_msgsnd = "{}/trace/msgs.snd.{}.csv".format(trace_dir, rank)
    rank_msgrcv = "{}/trace/msgs.rcv.{}.csv".format(trace_dir, rank)
    print(rank_msgcsv)

    log("Reading df")

    df = pd.read_csv(rank_msgcsv, sep="|")
    log("Df read")

    df = df.dropna().astype(
        {
            "rank": int,
            "peer": int,
            "timestep": int,
            "msg_id": int,
            "send_or_recv": int,
            "msg_sz": int,
            "timestamp": int,
        }
    )

    df_send = df[df["send_or_recv"] == 0]
    df_recv = df[df["send_or_recv"] == 1]

    df_send.to_csv(rank_msgsnd, index=None)
    df_recv.to_csv(rank_msgrcv, index=None)


def gen_prev_phase_df():
    phases_cur = ["FluxExchange", "BoundaryComm", "BoundaryApply", "LoadBalancing"]
    phases_prev = phases_cur[-1:] + phases_cur[:-1]
    prev_phase_df = pd.DataFrame({"phase_cur": phases_cur, "phase_prev": phases_prev})

    return prev_phase_df


#  @profile(precision=3)
def gen_phase_df(phase_map):
    prev_phase_df = gen_prev_phase_df()

    def mat_to_df(mat):
        ts, ranks = np.indices(mat.shape)

        col_ts = ts.ravel()
        col_ranks = ranks.ravel()
        col_data = mat.ravel()

        phase_df = pd.DataFrame(
            {"timestep": col_ts, "rank": col_ranks, "phase_time": col_data}
        )

        return phase_df

    all_phases = phase_map.keys()
    all_mat_dfs = []

    for phase in all_phases:
        phase_mat = phase_map[phase]
        mat_df = mat_to_df(phase_mat)
        mat_df["phase"] = phase

        if phase == "FluxExchange":
            mat_df["ts_prev"] = mat_df["timestep"] - 1
        else:
            mat_df["ts_prev"] = mat_df["timestep"]

        all_mat_dfs.append(mat_df)

    phase_df = pd.concat(all_mat_dfs)
    phase_df = phase_df.join(prev_phase_df.set_index("phase_cur"), on="phase")

    phase_df_cur = phase_df[["phase", "timestep", "rank", "phase_time"]]
    phase_df_prev = phase_df[["phase_prev", "ts_prev", "rank"]]

    phase_df_prev.columns = phase_df_cur.columns[:-1]
    phase_df_cur.set_index(["phase", "timestep", "rank"], inplace=True)
    phase_df_prev.set_index(["phase", "timestep", "rank"], inplace=True)
    tmp = phase_df_prev.join(phase_df_cur, on=["phase", "timestep", "rank"])
    # or
    # phase_df_prev = phase_df[cx, cy, cz].copy9)
    # phase_df_prev["phase_time"] = phase_df_cur["phase_time"] - will lookup by idx
    phase_df["phase_time_prev"] = tmp["phase_time"].to_numpy()
    phase_df.set_index(["phase", "timestep", "rank"], inplace=True)

    return phase_df


#  @profile(precision=3)
def aggr_msgs_map(rank, phase_df, trace_dir, max_ts):
    def log(msg):
        print("Rank {}: {}".format(rank, msg))

    rank_msgcsv = "{}/trace/msgs.rcv.{}.csv".format(trace_dir, rank)
    print(rank_msgcsv)

    t1 = time.time()
    log("Reading df")

    df = pd.read_csv(rank_msgcsv)
    df = df.dropna().astype(
        {
            "rank": int,
            "peer": int,
            "timestep": int,
            "msg_id": int,
            "send_or_recv": int,
            "msg_sz": int,
            "timestamp": int,
        }
    )

    t2 = time.time()
    log("Df read: {:.1f}s".format(t2 - t1))

    df = df[df["timestep"] < max_ts].copy()
    df = df.join(phase_df, on=["phase", "timestep", "peer"])
    df.rename(
        columns={
            "phase_time_prev": "peer_prevphasetime",
            "phase_time": "peer_curphasetime",
        },
        inplace=True,
    )

    t3 = time.time()
    log("Df joined: {:.1f}s".format(t3 - t2))

    # direct np.std(x) still produces NaNs with one val, idk why
    def std(x):
        return np.std(x)

    log("Aggregating df")

    df_agg = df.groupby(
        ["rank", "timestep", "phase", "send_or_recv"], as_index=False
    ).agg(
        {
            "msg_sz": ["mean", "sum", "count"],
            "peer": "nunique",
            "peer_prevphasetime": ["mean", std, "min", "max"],
            "peer_curphasetime": ["mean", std, "min", "max"],
        }
    )

    df_agg.columns = ["_".join(col).strip("_") for col in df_agg.columns.values]
    df_agg = df_agg.astype(
        {
            "msg_sz_mean": int,
            "peer_prevphasetime_mean": int,
            "peer_prevphasetime_std": int,
            "peer_curphasetime_mean": int,
            "peer_curphasetime_std": int,
        }
    )

    log("Writing df")
    df_agg.to_csv("{}/aggr2/msgs.{}.csv".format(trace_dir, rank), index=None)
    log("Done")


def combine_aggred_msgs(trace_dir):
    all_csvs = glob.glob(trace_dir + "/aggr2/msgs.*")
    #  all_csvs = all_csvs[:16]

    concat_csvpath = "{}/aggr/msg_concat.csv".format(trace_dir)

    all_dfs = None

    with multiprocessing.Pool(16) as p:
        all_dfs = p.map(pd.read_csv, all_csvs)

    df_concat = pd.concat(all_dfs)
    df_concat = (
        df_concat.sort_values(["timestep", "phase", "send_or_recv", "rank"])
        .groupby(["timestep", "phase", "send_or_recv"], as_index=False)
        .agg(
            {
                "rank": joinstr,
                "peer_nunique": joinstr,
                "msg_sz_mean": joinstr,
                "msg_sz_sum": joinstr,
                "msg_sz_count": joinstr,
                "peer_nunique": joinstr,
                "peer_prevphasetime_mean": joinstr,
                "peer_prevphasetime_std": joinstr,
                "peer_prevphasetime_min": joinstr,
                "peer_prevphasetime_max": joinstr,
                "peer_curphasetime_mean": joinstr,
                "peer_curphasetime_std": joinstr,
                "peer_curphasetime_min": joinstr,
                "peer_curphasetime_max": joinstr,
            }
        )
    )

    df_concat.to_csv(concat_csvpath, index=None)


def get_npsorted(d, s_or_r):
    arr = d[d["send_or_recv"] == s_or_r]["timestamp"].to_numpy()
    arr = np.sort(arr)
    return arr


@ray.remote
def read_msg_ts(args):
    trace_dir = args["trace_dir"]
    rank = args["rank"]
    hist_bins = args["hist"]
    timesteps = args["ts"]

    #  msgdf_path = "{}/msgs-shuffled/msgs.all.{}.csv".format(trace_dir, rank)
    msgdf_path = "{}/trace/msgs/msgs.{}.csv".format(trace_dir, rank)
    df = pd.read_csv(msgdf_path, sep="|")
    df = df[df["phase"] == "BoundaryComm"]
    #  df_aggr = df.groupby("timestep")

    all_snd = []
    all_rcv = []

    #  for gr_name, gr_df in df_aggr:
    for ts in timesteps:
        gr_df = df[df["timestep"] == ts]
        gr_snd = get_npsorted(gr_df, 0) / 1e3
        gr_rcv = get_npsorted(gr_df, 1) / 1e3

        gr_min_ts = np.min(gr_snd)
        gr_snd -= gr_min_ts
        gr_rcv -= gr_min_ts

        gr_snd = list(gr_snd)
        gr_rcv = list(gr_rcv)

        all_snd.append(gr_snd)
        all_rcv.append(gr_rcv)

    all_snd = [item for sublist in all_snd for item in sublist]
    all_rcv = [item for sublist in all_rcv for item in sublist]

    all_snd_hist, all_snd_bins = np.histogram(all_snd, bins=hist_bins)
    all_rcv_hist, all_rcv_bins = np.histogram(all_rcv, bins=hist_bins)

    return all_snd_hist, all_rcv_hist


class MsgTsReader(Task):
    nworkers = 4

    def __init__(self, trace_dir, timesteps):
        super().__init__(trace_dir)
        self.timesteps = timesteps

    def gen_worker_fn_args(self):
        args = super().gen_worker_fn_args()
        args["hist"] = list(range(0, 5000, 2))
        args["ts"] = self.timesteps
        return args

    @staticmethod
    def worker(args):
        return read_msg_ts(args)


def plot_msgtl(ts, bins, hist_snd, hist_rcv, plot_dir):
    ts_str = [str(i) for i in ts]
    ts_cs = ",".join(ts_str)
    ts_ps = ".".join(ts_str)

    fig, ax = plt.subplots(1, 1)

    plt.step(bins[:-1], hist_snd, label="Send")
    plt.step(bins[:-1], hist_rcv, label="Receive")

    ax.legend()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("# Messages")
    ax.set_title(
        "Send/Receive Latency Distribution (BoundaryComm, ts:{})".format(ts_cs)
    )

    ax.set_title("Message Latency Distribution (TS: 5K of 30K)")

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}K".format(x / 1e3))

    max_idx = np.max(np.nonzero(hist_rcv))
    max_bin = bins[max_idx]

    ax.plot([max_bin, max_bin], [0, 100000], linestyle="--", color="red")

    xlim_max = int(max_bin / 100) * 100 + 100
    xlim_max = 500

    ax.set_xlim([0, xlim_max])
    ax.set_ylim([0, 60000])

    fig.tight_layout()

    plot_path = "{}/msgtl_hist_ts{}.pdf".format(plot_dir, ts_ps)
    fig.savefig(plot_path, dpi=300)


def run_aggr_msgs():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    task = MsgAggrTask(trace_dir)
    task.cached = True
    task.npernode = 16
    task.nworkers = 2
    task.run_rankwise()
    #  combine_aggred_msgs(trace_dir)


def run_plot_msgtl():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"

    def run_plot_msgtl_ts(timesteps):
        mtr = MsgTsReader(trace_dir, timesteps)
        #  a, b = mtr.run_rank(0)
        all_hists = mtr.run_func_with_ray(read_msg_ts)
        hists_snd, hists_rcv = list(zip(*all_hists))
        hist_snd = np.sum(hists_snd, axis=0)
        hist_rcv = np.sum(hists_rcv, axis=0)
        bins = list(range(0, 5000, 2))

        plot_dir = "figures/20220809"
        plot_msgtl(timesteps, bins, hist_snd, hist_rcv, plot_dir)

    run_plot_msgtl_ts([1])
    run_plot_msgtl_ts([50])
    run_plot_msgtl_ts([500])
    run_plot_msgtl_ts([5000])
    run_plot_msgtl_ts([25000])


def plot_stats_senddf(send_df, chan_df):
    msg_df = join_msgs(send_df, chan_df)
    msg_df[msg_df["isflx"] == 0].groupby("ts").agg({"msgsz": "count"})
    msg_df[msg_df["isflx"] == 1].groupby("ts").agg({"msgsz": "count"})
    msg_df.columns
    aggr_msgdf = msg_df.groupby(["ts", "blk_id"]).agg(
        {"msgsz": ["mean", "std", "sum", "count"], "nbr_id": "nunique"}
    )

    aggr_msgdf = msg_df.groupby(["ts", "rank_x"]).agg(
        {"msgsz": ["mean", "std", "sum", "count"], "nbr_id": "nunique"}
    )

    sum_mat = aggr_msgdf["msgsz"]["sum"].unstack().values
    count_mat = aggr_msgdf["msgsz"]["count"].unstack().values
    nuniq_mat = aggr_msgdf["nbr_id"]["nunique"].unstack().values

    plot_imshow(sum_mat, "sum")
    count_mat = count_mat[2:, :]
    plot_imshow(count_mat, "count")
    nuniq_mat = nuniq_mat[2:, :]
    plot_imshow(nuniq_mat, "nuniq")

    t1 = chan_df[chan_df["ts"] == 1]
    ca = t1.groupby(["blk_id", "tag"]).agg({"nbr_id": ["min", "max", "count"]})

    aggr_df = send_df.groupby(
        [
            "rank",
            "ts",
        ],
        as_index=False,
    ).agg({"msgsz": ["mean", "std", "sum", "count"]})

    aggr_df.set_index(["rank", "ts"], inplace=True)

    all_stats = list(zip(*aggr_df.columns))[1]
    for stat in all_stats:
        print(f"Stat: {stat}")
        stat_mat = aggr_df["msgsz"][stat].unstack().values
        print(stat_mat)
        plot_imshow(stat_mat.T, stat)
        plot_stat_slice(stat_mat.T, stat, 6)

    pass


def run_regr_actual(X, y):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    print(regr.score(X, y))
    print(np.array(regr.coef_, dtype=int))
    print(np.array(regr.coef_))
    print(regr.intercept_)


def get_relevant_pprof_data(trace_name: str) -> Dict:
    kfls = "MultiStage_Step => Task_LoadAndSendFluxCorrections"
    kflr = "MultiStage_Step => Task_ReceiveFluxCorrections"
    kbcs = "MultiStage_Step => Task_LoadAndSendBoundBufs"
    kbcr = "MultiStage_Step => Task_ReceiveBoundBufs"
    mip = "MPI_Iprobe()"

    setup_tuple = setup_plot_stacked_generic(trace_name)
    stack_keys, stack_labels, ylim, ymaj, ymin = setup_tuple
    stack_keys = [kfls, kflr, kbcs, kbcr, mip]

    trace_dir = trace_dir_fmt.format(trace_name)
    concat_df = read_all_pprof_simple(trace_dir)
    pprof_data = filter_relevant_events(concat_df, stack_keys)

    data = {
        "kfls": pprof_data[kfls],
        "kflr": pprof_data[kflr],
        "kbcs": pprof_data[kbcs],
        "kbcr": pprof_data[kbcr],
        "mip": pprof_data[mip]
    }

    return data


def run_regr_wpprof():
    trace_name = "athenapk5"
    setup_tuple = setup_plot_stacked_generic(trace_name)
    stack_keys, stack_labels, ylim, ymaj, ymin = setup_tuple

    trace_dir = trace_dir_fmt.format(trace_name)
    concat_df = read_all_pprof_simple(trace_dir)
    pprof_data = filter_relevant_events(concat_df, stack_keys)

    for k in pprof_data.keys():
        print(k)

    chan_df, send_df = aggr_msgs_all("athenapk5")
    send_df = send_df[send_df["ts"] < 100]
    msg_df = join_msgs(send_df, chan_df)

    msg_df.columns
    msg_df[["Dest", "nbr_rank"]]
    msg_df[["blk_rank", "rank_x"]]
    msg_df = msg_df[msg_df["ts"] > 1]
    # only ts 0 and 1, load balancing etc Ig
    wtf_df = msg_df[msg_df["Dest"] != msg_df["nbr_rank"]]
    wtf_df = msg_df[msg_df["blk_rank"] != msg_df["rank_x"]]
    wtf_df

    flx_df = msg_df[msg_df["isflx"] == 1]
    flx_df
    nflx_df = msg_df[msg_df["isflx"] == 0]
    nflx_df

    groupby_cols = ["ts", "rank_x"]
    groupby_cols = ["ts", "nbr_rank"]
    flx_msgcnt_df = flx_df.groupby(groupby_cols, as_index=False).agg({"msgsz": "count"})

    bc_msgcnt_df = nflx_df.groupby(groupby_cols, as_index=False).agg({"msgsz": "count"})

    flxcnt_mat = (
        flx_msgcnt_df.pivot(index="ts", columns=groupby_cols[1], values="msgsz")
        .fillna(0)
        .to_numpy(dtype=int)
    )

    fdim = flxcnt_mat.shape
    flxmat = np.zeros((fdim[0], 512), dtype=int)
    flxmat[: fdim[0], : fdim[1]] = flxcnt_mat

    bccnt_mat = (
        bc_msgcnt_df.pivot(index="ts", columns=groupby_cols[1], values="msgsz")
        .fillna(0)
        .to_numpy(dtype=int)
    )

    fdim = bccnt_mat.shape
    bcmat = np.zeros((fdim[0], 512), dtype=int)
    bcmat[: fdim[0], : fdim[1]] = bccnt_mat

    kfls = "MultiStage_Step => Task_LoadAndSendFluxCorrections"
    kflr = "MultiStage_Step => Task_ReceiveFluxCorrections"
    kbcs = "MultiStage_Step => Task_LoadAndSendBoundBufs"
    kbcr = "MultiStage_Step => Task_ReceiveBoundBufs"

    # offset by 2 already
    x = flxmat[2]
    y = pprof_data[kflr]

    x = bcmat[2]
    y = pprof_data[kbcs]

    run_regr_actual(x.reshape(-1, 1), y)

    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    ax.plot(np.arange(512), pprof_data[kbcr], label="BC_Recv_Sec", zorder=2)
    ax2.plot(np.arange(512), bcmat[2], color="orange", label="BC_Recv_MsgCount")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    )
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    ax.set_title("Boundary Comm - Recv Time vs Recv Msg Count")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Total Time for Stage")
    ax2.set_ylabel("Num Messages")
    ax2.yaxis.set_label_position("right")
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.3, 0.06), ncol=2)
    PlotSaver.save(fig, trace_dir, None, "bc_recv")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax2 = ax.twinx()
    ax.plot(np.arange(512), pprof_data[kbcs], label="BC_Send_Sec", zorder=2)
    ax2.plot(np.arange(512), bcmat[2], color="orange", label="BC_Send_MsgCount")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f} s".format(x / 1e6))
    )
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    ax.set_title("Boundary Comm - Send Time vs Send Msg Count")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Total Time for Stage")
    ax2.set_ylabel("Num Messages")
    ax2.yaxis.set_label_position("right")
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.3, 0.06), ncol=2)
    PlotSaver.save(fig, trace_dir, None, "bc_send")

    pass


def run_aggr_msgs_new():
    global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    chan_df, send_df = aggr_msgs_all("athenapk13")


if __name__ == "__main__":
    #  run_aggr_msgs()
    #  run_plot_msgtl()
    #  plot_init()
    run_aggr_msgs_new()
