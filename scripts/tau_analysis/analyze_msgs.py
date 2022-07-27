import glob
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import subprocess
import sys
import time

from memory_profiler import profile

#  import ray
import traceback
from typing import Tuple

from trace_reader import TraceOps
from analyze_taskflow import get_exp_stats

from task import Task


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


def run_aggr_msgs():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    task = MsgAggrTask(trace_dir)
    task.cached = True
    task.npernode = 16
    task.nworkers = 2
    task.run_rankwise()
    #  combine_aggred_msgs(trace_dir)


if __name__ == "__main__":
    run_aggr_msgs()
