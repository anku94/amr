import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import subprocess
import sys
import time

from memory_profiler import profile

import ray
import traceback
from typing import Tuple

from trace_reader import TraceOps

from task import Task

ray.init(address="h0:6379")


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
    print(rank_msgcsv)

    df = pd.read_csv(rank_msgcsv, sep="|")
    #  print(df)
    #  print(df["phase"].unique())

    df = df.groupby(["rank", "timestep", "phase", "send_or_recv"], as_index=False).agg(
        {"msg_sz": ["mean", "sum", "count"], "peer": ["nunique", joinstr]}
    )

    df.columns = ["_".join(col).strip("_") for col in df.columns.values]

    df.to_csv("{}/aggr/msgs.{}.csv".format(trace_dir, rank), index=None)
    df = None


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


@profile(precision=3)
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


@profile(precision=3)
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

    ax.set_title(
        "Message Latency Distribution (TS: 5K of 30K)"
    )

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}K".format(x / 1e3))

    max_idx = np.max(np.nonzero(hist_rcv))
    max_bin = bins[max_idx]

    ax.plot([max_bin, max_bin], [0, 100000], linestyle='--', color='red')

    xlim_max = int(max_bin/100) * 100 + 100
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


if __name__ == "__main__":
    #  run_aggr_msgs()
    run_plot_msgtl()
