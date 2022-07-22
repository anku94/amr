import glob
import multiprocessing
import numpy as np
import pandas as pd
import subprocess
import sys
import ray
import traceback
from typing import Tuple, Dict

ray.init(address="h0:6379")


@ray.remote
def filter_func(args):
    tracedir = args["tracedir"]
    rank = args["rank"]

    dfin_path = "{}/trace/funcs.{}.csv".format(tracedir, rank)
    dfout_path = "{}/aggr/filtered.funcs.{}.csv".format(tracedir, rank)

    df_in = pd.read_csv(dfin_path, sep="|")
    df_in = df_in[df_in["func"] == "UpdateMeshBlockTree"]
    df_in.to_csv(dfout_path, index=None)

    return 0


def combine_filter_funcs(trace_dir):
    all_csvs = glob.glob(trace_dir + "/aggr/filtered.funcs.*")
    #  all_csvs = all_csvs[:16]

    concat_csvpath = "{}/aggr/filtfunc_concat.csv".format(trace_dir)

    all_dfs = None

    with multiprocessing.Pool(16) as p:
        all_dfs = p.map(pd.read_csv, all_csvs)

    def joinstr(x):
        return ",".join([str(i) for i in x])

    df_concat = pd.concat(all_dfs)
    df_concat = df_concat.astype(
        {
            "timestep": int,
            "timestamp": "int64",
            "enter_or_exit": int,
            "rank": int,
            "timestep": int,
        }
    )

    df_concat = (
        df_concat.sort_values(["timestep", "enter_or_exit", "rank"])
        .groupby(["timestep", "enter_or_exit"], as_index=False)
        .agg({"rank": joinstr, "timestamp": joinstr})
    )

    df_concat.to_csv(concat_csvpath, index=None)


def run_filter_funcs(tracedir) -> None:
    combine_filter_funcs(tracedir)
    return

    all_args = [{"tracedir": tracedir, "rank": r} for r in range(512)]

    remote_ret = [filter_func.remote(arg) for arg in all_args]
    result = ray.get(remote_ret)
    print(result)


def group_by_key(keystr, valstr) -> Dict:
    keys = keystr.split(",")
    vals = valstr.split(",")

    keys = [int(k) for k in keys]
    vals = [int(v) for v in vals]

    d = {}
    for k, v in zip(keys, vals):
        if k in d:
            d[k].append(v)
            d[k] = sorted(d[k])
        else:
            d[k] = [v]

    return d


def get_umbt_vars(keystr, valstr, nranks=512):
    rank_ts_map = group_by_key(keystr, valstr)

    val_lens = list(set(map(lambda x: len(x), rank_ts_map.values())))
    default_vals = [0] * val_lens[0]
    if len(val_lens) != 1:
        print(val_lens)
        assert False

    all_ts = []

    ranks_all = list(range(nranks))
    ranks_to_exclude = list(range(144, 160))
    ranks_filtered = [i for i in ranks_all if i not in ranks_to_exclude]
    for r in ranks_filtered:
        vals = default_vals
        if r in rank_ts_map:
            vals = rank_ts_map[r]

        assert len(vals) == val_lens[0]
        all_ts.append(vals)

    all_ts = list(zip(*all_ts))
    # just the first round
    all_ts = np.array(all_ts[0], dtype=np.int64)

    return all_ts


@ray.remote
def get_umbt_vars_wrapper(args):
    keystr = args["keystr"]
    valstr = args["valstr"]
    #  print(keystr, valstr)
    return get_umbt_vars(keystr, valstr)


def compute_deltas(ts_mat):
    # ts_mat is 30573*512
    p = ts_mat
    colavg = np.mean(ts_mat, axis=0)  # 512 ranks
    rowavg = np.mean(ts_mat, axis=1)  # 30000 ts

    # goal: minimize dist between rowavg and each col
    deltas = colavg - np.mean(rowavg)
    print(deltas)
    print(deltas.shape)
    return deltas


def analyze_deltas(deltas):
    num_nodes = int(len(deltas) / 16)
    deltas_split = np.split(deltas, num_nodes)
    all_vars = np.array([np.var(x) for x in deltas_split])
    all_stds = np.array(all_vars) ** 0.5

    mean_std = np.mean(all_stds)

    print(
        "[Delta Analysis] Num nodes: {:d}, Mean Std: {:.1f}".format(num_nodes, mean_std)
    )


def ray_analyze(df) -> None:
    all_args = zip(df["rank"], df["timestamp"])
    all_args = map(lambda x: {"keystr": x[0], "valstr": x[1]}, all_args)

    #  all_args = list(all_args)
    #  arg = all_args[0]
    #  print(get_umbt_vars_wrapper(arg))
    #  return

    remote_ret = [get_umbt_vars_wrapper.remote(arg) for arg in all_args]
    results = ray.get(remote_ret)
    results = np.array(results, dtype=np.int64)
    return results


def run_analyze(tracedir) -> None:
    umbt_path = "{}/aggr/filtfunc_concat.csv"
    df = pd.read_csv(umbt_path.format(tracedir))

    df_begin = df[df["enter_or_exit"] == 0]
    df_end = df[df["enter_or_exit"] == 1]

    tsbegin_mat = ray_analyze(df_begin)
    tsend_mat = ray_analyze(df_end)

    deltas = compute_deltas(tsend_mat)
    analyze_deltas(deltas)

    tsbegin_mat = tsbegin_mat + deltas
    tsend_mat = tsend_mat + deltas

    def analyze(mat):
        mat_var = np.var(mat, axis=1)
        var_mean = np.mean(mat_var)

        print("Mean Var: {:.2f}".format(var_mean))

        mat_max = np.max(mat, axis=1)
        mat_min = np.min(mat, axis=1)
        mat_delta = mat_max - mat_min
        mat_delta_mean = np.mean(mat_delta)

        print("Mean Lifespan: {:.2f}".format(mat_delta_mean))

    analyze(tsbegin_mat)
    analyze(tsend_mat)

    tsdelta_mat = tsend_mat - tsbegin_mat
    tsdelta_max = np.max(tsdelta_mat, axis=1)
    tsdelta_min = np.min(tsdelta_mat, axis=1)

    print(
        "Timestamp Delta, Avg min: {:.2f}, avg max: {:.2f}".format(
            np.mean(tsdelta_min), np.mean(tsdelta_max)
        )
    )

    print(tsdelta_min)


@ray.remote
def analyze_state(args):
    tracedir = args["tracedir"]
    rank = args["rank"]

    df_path = "{}/trace/state.{}.csv".format(tracedir, rank)
    df = pd.read_csv(df_path, sep="|")

    tc = df[df["key"] == "TC"]["val"].tolist()
    cl = df[df["key"] == "CL"]["val"].tolist()
    rl = df[df["key"] == "RL"]["val"].tolist()

    def parse_ls(ls, t):
        return [t(i) for i in ls.strip(",").split(",")]

    def assert_ones(s):
        s = parse_ls(s, float)
        for val in s:
            assert int(val) == 1

    for cost in cl:
        assert_ones(cost)

    ranks = [parse_ls(rl_i, int) for rl_i in rl]
    rank_tc = [len(r)/512.0 for r in ranks]
    target_costs = [float(i) for i in tc]

    a = np.array(rank_tc)
    b = np.array(target_costs)
    assert sum(a - b) < 1e-1

    return target_costs


def run_state() -> None:
    tracedir = "/mnt/ltio/parthenon-topo/profile8"
    num_ranks = 512

    all_args = [{"tracedir": tracedir, "rank": r} for r in range(num_ranks)]
    args = all_args[0]
    analyze_state(args)
    remote_ret = [analyze_state.remote(arg) for arg in all_args]
    result = ray.get(remote_ret)
    tc_mat = np.array(result)
    sum_var = sum(np.var(tc_mat, axis=0))
    assert sum_var < 1e-1


def run() -> None:
    tracedir = "/mnt/ltio/parthenon-topo/profile8"
    #  run_filter_funcs(tracedir)
    run_analyze(tracedir)


if __name__ == "__main__":
    run()
    run_state()
