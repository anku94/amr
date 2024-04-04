import glob
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pathlib
import pickle
import re
import subprocess
import struct
import sys
import time
import os

try:
    import ipdb
    import ray
    from task import Task, ProfOutputReader
except Exception as e:
    pass
import traceback

from common import plot_init_big as plot_init, PlotSaver, prof_evt_map
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import List, Tuple
from pathlib import Path
from trace_reader import TraceReader, TraceOps
from time import time

global npernode
npernode = 16

global nworkers
nworkers = 16


def decorator_timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time() - t1
        return result, end
        pass

    return wrapper


def get_prof_path(trace_dir: str, evt: int) -> str:
    # replace merged with agg if single self-contained run
    ppath = f"{trace_dir}/prof.merged.evt{evt}.csv"
    ppath = f"{trace_dir}/prof.aggr.evt{evt}.csv"
    return ppath


def get_prof_path_rw(trace_dir: str, evt: int, rbeg: int, rend: int) -> str:
    path_base = get_prof_path(trace_dir, evt)
    ps = pathlib.Path(path_base).suffix
    pp = path_base[: -len(ps)]
    return f"{pp}.{rbeg}-{rend}{ps}"


def read_evt_aggr(evt: int) -> pd.DataFrame:
    global trace_dir
    df_path = get_prof_path(trace_dir, evt)
    df = pd.read_csv(df_path)
    df.sort_values(["sub_ts", "rank", "block_id"], inplace=True)
    aggr_df = df.groupby("sub_ts", as_index=False).agg(
        {"rank": list, "block_id": list, "time_us": list}
    )


def rsbe_group_func(df):
    df_agg = df.groupby(["sub_ts", "block_id"], as_index=False).agg({"data": list})

    return df_agg


def rsbe_add_func(df):
    df_agg = df.groupby(["sub_ts", "block_id"], as_index=False).agg({"data": "sum"})
    # df_agg.set_index(["sub_ts", "block_id"], inplace=True)
    return df_agg


"""
More efficient version of run_sep_by_evt
"""


def rsbe_aggr_by_rank(evt_code: int, rbeg: int, rend: int, nranks: int):
    global npernode, nworkers

    #  evt_code = 0
    reader = ProfOutputReader(trace_dir, evt_code, nranks, npernode, nworkers)
    all_dfs = reader.run_rankwise(rbeg, rend)

    # aggregate in advance for lower mem footprint
    with multiprocessing.Pool(16) as p:
        #  agg_dfs = p.map(rsbe_add_func, all_dfs)
        agg_dfs = p.map(rsbe_add_func, all_dfs)

    aggr_df = pd.concat(agg_dfs)

    del agg_dfs
    del all_dfs
    return aggr_df


def run_sep_by_evt_eff(evt_code):
    rbeg, rend = 0, 16

    global trace_dir
    tr = TraceReader(trace_dir)
    reader = ProfOutputReader(trace_dir, evt_code)
    all_dfs = reader.run_rankwise(rbeg, rend)

    aggr_df = pd.concat(all_dfs)
    del all_dfs

    inter_out = f"{trace_dir}/aggr.inter.{rbeg}-{rend}.evt{evt_code}.csv"
    inter_out = f"{trace_dir}/aggr.evt{evt_code}.inter.{rbeg}-{rend}.csv"
    inter_out
    aggr_df
    aggr_df.to_csv(inter_out, index=None)
    pass


def run_sep_by_evt_create_mat_eff(evt_code: int, nranks: int):
    #  allr = np.arange(0, nranks + 1, 256)
    #  rpairs = list(zip(allr[:-1], allr[1:]))

    #  all_dfs = list(map(lambda x: rsbe_aggr_by_rank(evt_code, *x, nranks), rpairs))
    #  aggr_df = pd.concat(all_dfs)

    #  del all_dfs
    aggr_df_fpath = f"{trace_dir}/aggr.tmp.evt{evt_code}.csv"
    #  aggr_df.to_csv(aggr_df_fpath, index=None)
    aggr_df = pd.read_csv(aggr_df_fpath)

    pivot_table = aggr_df.pivot_table(
        values="data", index="sub_ts", columns="block_id", aggfunc=np.sum
    )
    #  pivot_table.to_numpy().shape
    mat = pivot_table.to_numpy()
    mat_npy = f"{trace_dir}/evt{evt_code}.mat.npy"
    np.save(mat_npy, mat)
    return

    mat = [list(filter(lambda x: not np.isnan(x), row)) for row in mat]

    mat_file = f"{trace_dir}/evt{evt_code}.mat.pickle"
    print(f"Writing to {mat_file}")

    with open(mat_file, "wb") as f:
        f.write(pickle.dumps(mat))

    mat = pickle.loads(open(mat_file, "rb").read())
    print(f"Event {evt_code}: matrix verified: {len(mat)}, {len(mat[0])}")
    pass


#  evt_code = 1
def run_sep_by_evt_create_mat(evt_code, nranks):
    global trace_dir, npernode, nworkers
    tr = TraceReader(trace_dir)
    reader = ProfOutputReader(trace_dir, evt_code, nranks, npernode, nworkers)
    #  all_dfs = reader.run_rankwise(0, 512)
    #  nranks = 64
    # nranks = 1024
    all_dfs = reader.run_rankwise(0, nranks)

    aggr_df = pd.concat(all_dfs)
    #  print(evt_code, aggr_df)

    print(f"aggr_df for evt: {evt_code}: {aggr_df}")

    pivot_table = aggr_df.pivot_table(
        values="data", index="sub_ts", columns="block_id", aggfunc=np.sum
    )
    pivot_table.to_numpy().shape
    mat = pivot_table.to_numpy()

    #  len(pivot_table.iloc[0, :].dropna().to_list())
    #  len(np.sum(~np.isnan(mat), axis=1))
    mat = [list(filter(lambda x: not np.isnan(x), row)) for row in mat]

    mat_file = f"{trace_dir}/evt{evt_code}.mat.pickle"
    print(f"Writing to {mat_file}")

    with open(mat_file, "wb") as f:
        f.write(pickle.dumps(mat))

    mat = pickle.loads(open(mat_file, "rb").read())
    print(f"Event {evt_code}: matrix verified: {len(mat)}, {len(mat[0])}")
    return

    #  aggr_df

    #  with multiprocessing.Pool(16) as p:
        #  agg_dfs = p.map(rsbe_group_func, all_dfs)

    # aggregate in advance for lower mem footprint

    with multiprocessing.Pool(16) as p:
        agg_dfs = p.map(rsbe_add_func, all_dfs)

    aggr_df = pd.concat(agg_dfs)
    aggr_df.pivot
    aggr_df["dlen"] = aggr_df["data"].apply(lambda x: len(x))

    print(f"Value counts per block-id per ts, for evt {evt_code}:")
    print(aggr_df["dlen"].value_counts())

    aggr_df.sort_values(["sub_ts", "block_id"], inplace=True)
    ag2_df = aggr_df.groupby("sub_ts", as_index=False).agg(
        {"block_id": list, "data": list}
    )

    mat = ag2_df["data"].to_list()

    mat_file = f"{trace_dir}/evt{evt_code}.mat.pickle"
    print(f"Writing to {mat_file}")

    with open(mat_file, "wb") as f:
        f.write(pickle.dumps(mat))

    mat = pickle.loads(open(mat_file, "rb").read())
    print(f"Event {evt_code}: matrix verified: {len(mat)}, {len(mat[0])}")


def run_sep_by_evt_inner(evt, rbeg, rend, evt_df_path, nranks: int):
    global trace_dir, npernode, nworkers
    tr = TraceReader(trace_dir)
    reader = ProfOutputReader(trace_dir, evt, nranks, npernode, nworkers)
    all_dfs = reader.run_rankwise(rbeg, rend)
    aggr_df = pd.concat(all_dfs)

    #  aggr_df.sort_values(["ts", "sub_ts", "rank", "block_id"], inplace=True)
    # below added later to make sure analysis of actual isn't buggy
    aggr_df.sort_values(["sub_ts", "block_id"], inplace=True)

    col_names = [
        "time_us",
        "time_us",
        "refine_flag",
        "block_idx",
        "cost",
        "time_us",
        "time_us",
    ]
    cols_new = list(aggr_df.columns)[:-1]
    cols_new.append(col_names[evt])
    aggr_df.columns = cols_new

    if evt == 4:
        aggr_df["sub_ts"] -= 1
        #  aggr_df["cost"] *= 1000

    print(f"Writing evt {evt} to {evt_df_path}...")
    print(aggr_df)
    aggr_df.to_csv(evt_df_path, index=None)


def run_sep_by_evt_util(evt, nranks: int):
    global trace_dir
    rbase = np.arange(0, nranks + 1, 64)
    #  rbase = np.arange(0, 65, 64)
    all_rbeg = rbase[:-1]
    all_rend = rbase[1:]

    all_pairs = list(zip(all_rbeg, all_rend))
    all_tmp_dfs = []

    for rbeg, rend in all_pairs:
        print(f"Extracting rank pair {rbeg}-{rend}")
        df_out = get_prof_path_rw(trace_dir, evt, rbeg, rend)
        all_tmp_dfs.append(df_out)
        run_sep_by_evt_inner(evt, rbeg, rend, df_out, nranks)

    all_dfs = map(pd.read_csv, all_tmp_dfs)
    df_merged = pd.concat(all_dfs)

    df_merged_out = get_prof_path(trace_dir, evt)
    df_merged.to_csv(df_merged_out, index=None)
    pass


def run_sep_by_evt(nranks: int):  #  evts = [0, 1, 3, 4, 5, 6] evts = [3]
    #  evts = [0, 1]
    #  for evt in evts:
        #  run_sep_by_evt_create_mat_eff(evt, nranks)

    evts = [2, 3]
    for evt in evts:
        run_sep_by_evt_util(evt, nranks)


def read_aggr_more(evt_code):
    df_path = f"{trace_dir}/prof.aggrmore.evt{evt_code}.csv"
    df = pd.read_csv(df_path)

    if evt_code == 2:
        cols = ["block_id", "refine_flag"]
    else:
        cols = ["block_id"]

    for col in cols:
        df[col] = safe_ls(df[col])

    return df


def run_aggr_costs_evt(evt_code):
    global trace_dir
    df_path = f"{trace_dir}/prof.aggr.evt{evt_code}.csv"
    df = pd.read_csv(df_path)
    aggr_df = df.groupby(["sub_ts", "block_id"], as_index=False).agg(
        {"rank": "min", "time_us": "sum"}
    )

    aggr_df.sort_values(["sub_ts", "block_id"], inplace=True)

    #  aggr_df2 = aggr_df.groupby([ "sub_ts" ], as_index=False).agg({
    #  "block_id": list,
    #  "rank": "min",
    #  "time_us": list
    #  })

    return aggr_df


def run_aggr_costs():
    global trace_dir
    df0 = run_aggr_costs_evt(0)
    df1 = run_aggr_costs_evt(1)

    aggr_df = df0.merge(df1, on=["sub_ts", "block_id"], how="outer", sort=True)

    na_vals = {
        "rank_x": -1,
        "time_us_x": 0,
        "rank_y": -1,
        "time_us_y": 0,
    }

    ag2_df = aggr_df.fillna(value=na_vals).astype(
        {
            "rank_x": int,
            "time_us_x": int,
            "rank_y": int,
            "time_us_y": int,
        }
    )

    ag2_df["time_us"] = ag2_df["time_us_x"] + ag2_df["time_us_y"]
    ag2_df.sort_values(["sub_ts", "block_id"], inplace=True)
    ag3_df = ag2_df.groupby(["sub_ts"], as_index=False).agg({"time_us": list})

    df_out_path = f"{trace_dir}/prof.aggrmore.evt01.csv"
    ag3_df.to_csv(df_out_path, index=None)


def run_aggr_assns():
    evt_code = 3
    global trace_dir
    df_path = f"{trace_dir}/prof.aggr.evt{evt_code}.csv"
    df_out_path = f"{trace_dir}/prof.aggrmore.evt{evt_code}.csv"

    #  if os.path.exists(df_out_path):
    #  print(f"File {df_out_path} exists. Just reading that")
    #  return read_aggr_more(evt_code)

    df = pd.read_csv(df_path)

    nranks = df["rank"].max() + 1

    all_dfs = []

    for rank in range(nranks):
        ref_df = df[df["rank"] == rank].copy()
        ref_df["sub_ts"] = ref_df["block_idx"].eq(0).cumsum() - 1
        all_dfs.append(ref_df)

    del df
    concat_df = pd.concat(all_dfs)

    aggr_df = concat_df.groupby("sub_ts", as_index=False).agg(
        {"ts": "min", "rank": list, "block_id": list}
    )

    aggr_df.to_csv(df_out_path, index=None)

    return aggr_df


def run_aggr_refs(nranks: int):
    global npernode, nworkers

    evt_code = 2
    global trace_dir
    df_out_path = f"{trace_dir}/prof.aggrmore.evt{evt_code}.csv"

    #  if os.path.exists(df_out_path):
    #  print(f"File {df_out_path} exists. Just reading that")
    #  return read_aggr_more(evt_code)

    reader = ProfOutputReader(trace_dir, evt_code, nranks, npernode, nworkers)
    df0 = reader.run_rank(0)
    print(df0)

    cols = list(df0.columns)
    cols[-1] = "refine_flag"
    df0.columns = cols

    ref_df = df0.copy()
    ref_df["sub_ts"] = ref_df["block_id"].eq(-1).cumsum()
    ref_df = ref_df[ref_df["block_id"] != -1]

    aggr_df = ref_df.groupby("sub_ts", as_index=False).agg(
        {"ts": "min", "block_id": list, "refine_flag": list}
    )

    #  aggr_df["nblocks"] = aggr_df["refine_flag"].apply(lambda x: len(x))
    aggr_df["nref"] = aggr_df["refine_flag"].apply(lambda x: x.count(1))
    aggr_df["nderef"] = aggr_df["refine_flag"].apply(lambda x: x.count(-1))

    #  aggr_df["sub_ts"] += 1

    aggr_df.to_csv(df_out_path, index=None)
    # TODO: write function to persist aggr_df, here and the other one
    # in cpp-readable format
    return aggr_df


def run_aggr_costs():
    df.sort_values(["sub_ts", "block_id"], inplace=True)

    aggr_df = df.groupby(["sub_ts"], as_index=False).agg(
        {"ts": "unique", "rank": list, "block_id": list, "cost": list}
    )
    pass


def validate_refinements(blk_df: pd.DataFrame, ref_df: pd.DataFrame) -> None:
    blk_df["nblocks"] = blk_df["block_id"].apply(lambda x: len(x))

    blk_df["nblocks_cur_ts"] = blk_df["nblocks"]
    blk_df["nblocks_next_ts"] = blk_df["nblocks"].shift(-1, fill_value=-1)
    # no next_ts for last row
    blk_df = blk_df.iloc[:-1, :]

    merged_df = blk_df.merge(ref_df, how="left", on="sub_ts").fillna(
        0, downcast="infer"
    )

    tmp_df = merged_df[["nblocks_cur_ts", "nref", "nderef", "nblocks_next_ts"]].copy()

    # for 3D code
    #  tmp_df["nblocks_next_ts_computed"] = (
        #  tmp_df["nblocks_cur_ts"] + 7 * tmp_df["nref"] - 7 * tmp_df["nderef"] / 8
    #  ).astype(int)

    # for 2D code
    tmp_df["nblocks_next_ts_computed"] = (
        tmp_df["nblocks_cur_ts"] + 3 * tmp_df["nref"] - 3 * tmp_df["nderef"] / 4
    ).astype(int)

    try:
        assert (tmp_df["nblocks_next_ts_computed"] == tmp_df["nblocks_next_ts"]).all()
        print("Refinements validated. All block counts as expected!!")
    except AssertionError as e:
        mismatch_df = tmp_df[
            tmp_df["nblocks_next_ts_computed"] != tmp_df["nblocks_next_ts"]
        ]
        print(mismatch_df)
        return


def get_evtmat_by_bid(trace_dir: str, evt_code: int) -> Tuple[np.ndarray, np.ndarray]:
    nblocks_npy = f"{trace_dir}/nblocks.bid.{evt_code}.npy"
    evtmat_npy = f"{trace_dir}/evtmat.bid.{evt_code}.npy"

    if os.path.exists(nblocks_npy) and os.path.exists(evtmat_npy):
        with open(nblocks_npy, "rb") as f:
            nblocks = np.load(f)

        with open(evtmat_npy, "rb") as f:
            time_evtmat = np.load(f)

        print(f"Reading {trace_dir} evtmat from cache. Shape: {time_evtmat.shape}")
        return nblocks, time_evtmat

    df_path = get_prof_path(trace_dir, evt_code)
    df = pd.read_csv(df_path)
    df.sort_values(["sub_ts", "block_id"], inplace=True)
    aggr_df = df.groupby("sub_ts", as_index=False).agg(
        {"ts": "unique", "rank": "unique", "block_id": list, "time_us": list}
    )

    nblocks = aggr_df["time_us"].apply(lambda x: len(x)).to_numpy()
    nbmax = max(nblocks)

    time_us_std = aggr_df["time_us"].apply(
        lambda x: np.array(x + [0] * (nbmax - len(x)))
    )
    time_evtmat = np.stack(time_us_std)

    with open(nblocks_npy, "wb") as f:
        np.save(f, nblocks)

    with open(evtmat_npy, "wb") as f:
        np.save(f, time_evtmat)

    return nblocks, time_evtmat


"""
Returns a rank-wise matrix for a kernel
"""


def read_times(evt_code: int) -> pd.DataFrame:
    global trace_dir
    df_path = get_prof_path(trace_dir, evt_code)
    df = pd.read_csv(df_path)
    df.sort_values(["sub_ts", "block_id"], inplace=True)

    nblocks = df.groupby("sub_ts").agg({"block_id": "nunique"})
    nblocks = nblocks["block_id"].reset_index()["block_id"]

    prof_df = df.groupby(["sub_ts", "rank"], as_index=False).agg({"time_us": "sum"})

    all_sub_ts = prof_df["sub_ts"].unique()
    nranks = prof_df["rank"].max() + 1

    all_rank_tuples = []

    for ts in np.arange(-1, all_sub_ts[-1] + 1):
        if ts not in all_sub_ts:
            for rank in np.arange(0, nranks):
                rank_tuple = (ts, rank, 0)
                all_rank_tuples.append(rank_tuple)

    missing_df = pd.DataFrame(all_rank_tuples, columns=prof_df.columns)
    final_df = pd.concat([prof_df, missing_df])
    final_df.sort_values(["sub_ts", "rank"], inplace=True)
    prof_mat = final_df["time_us"].to_numpy().reshape((-1, nranks))

    return nblocks, prof_mat


def validate_nblocks(blk_df: pd.DataFrame, nblocks_prof: pd.Series) -> None:
    nblocks_blk = blk_df["block_id"].apply(lambda x: len(x))
    #  nblocks_prof = prof_df["time_us"].apply(lambda x: len(x))

    print(f"Validating times for event against assignments")
    if nblocks_blk.size == nblocks_prof.size:
        print("\tSizes match. Promising start")
        if (nblocks_blk == nblocks_prof).all():
            print("\tAll nblocks match")
        else:
            print("\t ERROR. Nblocks do not match. Uncertain how to proceed.")
    else:
        print(f"\t ERROR. Size mismatch. {nblocks_blk.size} vs {nblocks_prof.size}")


def compute_prof_mat_stats(prof_mat):
    print("Computing misc stats on prof_mat")

    total_rankhours = prof_mat.sum() / (1e6 * 3600)
    min_runtime = prof_mat.mean(axis=1).sum() / 1e6
    actual_runtime = prof_mat.max(axis=1).sum() / 1e6

    print("\tTotal RankHours: {:.1f}s".format(total_rankhours))
    print("\tMin Runtime Possible: {:.1f}s".format(min_runtime))
    print("\tActual Runtime: {:.1f}s".format(actual_runtime))

    pass


def analyze_prof_mat(prof_mat):
    pass


def safe_ls(ls):
    if type(ls) == str:
        ls = ls.strip("[]")
        ls = ls.split(",")
        ls = [int(i) for i in ls]
        return ls

    return ls


def divide_ref(block_ids, ref_flags):
    block_ids = safe_ls(block_ids)
    ref_flags = safe_ls(ref_flags)

    try:
        assert len(block_ids) == len(ref_flags)
    except AssertionError as e:
        print(block_ids, ref_flags)
        print(len(block_ids), len(ref_flags))
        raise AssertionError("idk")

    blocks_ref = []
    blocks_deref = []

    for idx, bid in enumerate(block_ids):
        if ref_flags[idx] == 1:
            blocks_ref.append(bid)
        elif ref_flags[idx] == -1:
            blocks_deref.append(bid)

    return blocks_ref, blocks_deref


def write_refinements():
    global trace_dir
    df_path = f"{trace_dir}/prof.aggrmore.evt2.csv"
    df_out = f"{trace_dir}/refinements.bin"

    df = pd.read_csv(df_path)

    write_int = lambda f, i: f.write(struct.pack("i", i))

    with open(df_out, "wb") as f:
        print(f"Writing to {df_out}")
        for ridx, row in df.iterrows():
            write_int(f, row["ts"])
            write_int(f, row["sub_ts"])

            bids = row["block_id"]
            refs = row["refine_flag"]
            bl_ref, bl_deref = divide_ref(bids, refs)

            write_int(f, len(bl_ref))
            for bid in bl_ref:
                write_int(f, bid)

            write_int(f, len(bl_deref))
            for bid in bl_deref:
                write_int(f, bid)

    return


def write_assignments():
    df_path = f"{trace_dir}/prof.aggrmore.evt3.csv"
    df_out = f"{trace_dir}/assignments.bin"

    df = pd.read_csv(df_path)

    write_int = lambda f, i: f.write(struct.pack("i", i))

    with open(df_out, "wb") as f:
        print(f"Writing to {df_out}")
        for ridx, row in df.iterrows():
            write_int(f, row["ts"])
            write_int(f, row["sub_ts"])

            ranks = safe_ls(row["rank"])
            bids = safe_ls(row["block_id"])

            assert len(ranks) == len(bids)

            ranks_bids = list(zip(ranks, bids))
            ranks_bids = sorted(ranks_bids, key=lambda x: x[1])
            ranks_ordby_bids = list(map(lambda x: x[0], ranks_bids))

            write_int(f, len(ranks_ordby_bids))

            for i in ranks_ordby_bids:
                write_int(f, i)

    return


def join_two_traces(trace_a, trace_b, evt_code):
    df_a = pd.read_csv(f"{trace_a}/prof.aggr.evt{evt_code}.csv")
    df_b = pd.read_csv(f"{trace_b}/prof.aggr.evt{evt_code}.csv")

    total_ts = 30000

    amin = df_a["ts"].iloc[0]
    amax = df_a["ts"].iloc[-1]
    alen = len(df_a)

    bmin = df_b["ts"].iloc[0]
    bmax = df_b["ts"].iloc[-1]
    blen = len(df_b)

    delta_b = total_ts - bmax - 1
    df_b["ts"] += delta_b

    df_bt = df_b[df_b["ts"] > amax]

    df_merged = pd.concat([df_a, df_bt])
    df_merged.to_csv(f"{trace_b}/prof.merged.evt{evt_code}.csv", index=None)


def run_join_two_traces():
    trace_a = "/mnt/ltio/parthenon-topo/profile19"
    trace_b = "/mnt/ltio/parthenon-topo/profile20"
    join_two_traces(trace_a, trace_b, "1")


def plot_aggr_stats(evt: int, smooth: int):
    evt_label = prof_evt_map[evt]
    fname = f"prof.aggrstats.evt{evt}.smooth{smooth}"

    df_a = pd.read_csv(f"{trace_dir}/prof.merged.evt{evt}.csv")
    df_aggr = df_a.groupby("ts", as_index=False).agg(
        {"time_us": ["min", "max", "mean"]}
    )

    data_x = df_aggr["ts"]
    data_min = df_aggr["time_us"]["min"]
    data_max = df_aggr["time_us"]["max"]
    data_mean = df_aggr["time_us"]["mean"]

    if smooth > 0:
        data_min = TraceOps.smoothen_1d(data_min)
        data_max = TraceOps.smoothen_1d(data_max)
        data_mean = TraceOps.smoothen_1d(data_mean)

    fig, ax = plt.subplots(1, 1)

    ax.plot(data_x, data_min, label="Min")
    ax.plot(data_x, data_max, label="Max")
    ax.plot(data_x, data_mean, label="Mean")

    ax.set_title(f"Aggr Stats: {evt_label} (smooth={smooth})")
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Time (ms)")

    ax.legend()

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))
    ax.set_ylim([0, 50000])

    ax.yaxis.set_major_locator(MultipleLocator(5000))
    ax.yaxis.set_minor_locator(MultipleLocator(1000))
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")

    fig.tight_layout()
    PlotSaver.save(fig, trace_dir, None, fname)


def plot_rankhours(evts):
    get_rh = lambda evt: pd.read_csv(f"{trace_dir}/prof.merged.evt{evt}.csv")[
        "ts"
    ].sum()

    rh_us = np.array(list(map(get_rh, evts)))
    rh_hours = rh_us / (1e6 * 3600)

    fig, ax = plt.subplots(1, 1)

    ax.bar(np.array(evts), rh_hours, width=0.5)
    ax.set_xticks(evts, prof_evt_map)
    #  ax.set_xticklabels(prof_evt_map)
    ax.set_xlabel("Profiled Event Name")
    ax.set_ylabel("Rank-Hours")
    ax.set_ylim([0, 800])

    ax.yaxis.grid(which="major", color="#bbb")
    ax.set_axisbelow(True)

    PlotSaver.save(fig, trace_dir, None, "prof.rankhours")


def _aggr_nparr_roundrobin(np_arr, nout):
    all_isums = []
    for i in range(nout):
        slice_i = np_arr[:, i::nout]
        sum_i = slice_i.sum(axis=1)
        all_isums.append(sum_i)

    all_isum_nparr = np.array(all_isums).T
    return all_isum_nparr


def _aggr_nparr_by_rank(np_arr, nranks):
    aggr_arr = _aggr_nparr_roundrobin(np_arr, nranks)
    sum_a = np_arr.sum(axis=1)
    sum_b = aggr_arr.sum(axis=1)
    assert (sum_a == sum_b).all()
    return aggr_arr


"""
Does a round-robin aggregation of array rank-wise,
and then a contiguous aggregation node-wise.

This assumes that the meshblock allocation scheme is round-robin,
which was true when this code was written, but will probably change later.

Then we'll have to consult some allocation map to aggregate.
"""


def aggr_block_nparr(np_arr, nranks, nnodes) -> Tuple[np.array, np.array]:
    assert nranks % nnodes == 0
    arr_rankwise = _aggr_nparr_by_rank(np_arr, nranks)
    npernode = int(nranks / nnodes)

    all_nsums = []
    for n in range(nnodes):
        i = n * npernode
        j = i + npernode
        node_sum = arr_rankwise[:, i:j].sum(axis=1)
        all_nsums.append(node_sum)

    all_nsums = np.array(all_nsums).T
    assert (all_nsums.sum(axis=1) == np_arr.sum(axis=1)).all()
    arr_nodewise = all_nsums
    return arr_rankwise, arr_nodewise


def _read_and_reg_evt(evt: str, clip=None):
    global trace_dir
    df_path = get_prof_path(trace_dir, evt)
    print(f"Reading dataframe: {df_path}")

    df0_agg = df0.groupby("sub_ts", as_index=False).agg({"time_us": list})
    df1_agg = df1.groupby("sub_ts", as_index=False).agg({"time_us": list})

    df = pd.read_csv(df_path)
    df_agg = df.groupby("sub_ts", as_index=False).agg({"time_us": list})
    df_agg = df_agg[df_agg["sub_ts"] >= 2]

    bw_1d = df["time_us"].to_numpy()
    bw_2d = df_agg["time_us"].to_list()

    bw_1d_rel = []
    bw_2d_rel = []

    for row in bw_2d:
        nprow = np.array(row)
        nprow_rel = nprow - np.min(nprow)
        bw_2d_rel.append(nprow_rel)
        bw_1d_rel += list(nprow_rel)

    bw_2d = TraceOps.uniform_2d_nparr(bw_2d)
    bw_2d_rel = TraceOps.uniform_2d_nparr(bw_2d_rel)

    if clip is not None:
        bw_2d = bw_2d[:, :clip]
        bw_2d_rel = bw_2d_rel[:, :clip]

    return (bw_1d, bw_2d, bw_1d_rel, bw_2d_rel)


def _add_tuples(evt_tuples: List):
    # 1d_abs, 2d_abs, 1d_rel, 2d_rel

    a, b, c, d = evt_tuples[0]
    for evt_tuple in evt_tuples[1:]:
        at, bt, ct, dt = evt_tuple
        try:
            a += at
            c += ct
        except Exception as e:
            a = None
            c = None

        try:
            b += bt
        except Exception as e:
            minb = min(b.shape[1], bt.shape[1])
            print(f"Clipping timegrids during addition: {b.shape} and {bt.shape}")

            b = b[:, :minb]
            bt = bt[:, :minb]
            b += bt

    # d is relative, needs to be computed from b
    d = _get_relative_timegrid(b)
    return (a, b, c, d)


def _get_relative_timegrid(data_timegrid: np.array) -> np.array:
    d = data_timegrid
    dm = data_timegrid.min(axis=1)
    return (d.T - dm).T


def _fig_make_cax(fig):
    # left and right boundaries of the subplot area
    fig.subplots_adjust(left=0.15, right=0.78)
    # left, bottom, width, height
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])
    return cax


def _evt_get_labels(evt: str, proftype: str, diff_mode: bool) -> Tuple[str, str]:
    evt_name = prof_evt_map[evt]
    evt_abbrev = "".join(re.findall(r"[A-Z\+]", evt_name))
    evt_abbrev_fsafe = evt_abbrev.replace("+", "_").lower()

    rel = "rel." if diff_mode else ""
    rel_title = " (rel) " if diff_mode else ""

    plot_fname = f"prof.timegrid.{rel}{proftype}wise.{evt_abbrev_fsafe}"
    plot_title = f"{proftype.title()}wise times{rel_title}for event: {evt_abbrev}"

    print("Plot fname: ", plot_fname)
    print("Plot title: ", plot_title)
    return plot_fname, plot_title


def plot_timegrid_blockwise( evt: str, data_blockwise: np.array, data_1d: np.array, diff_mode=False):
    plot_fname, title = _evt_get_labels(evt, "block", diff_mode=diff_mode)

    if data_1d is None:
        data_1d = data_blockwise.flatten()
        data_1d = data_1d[data_1d > 0]

    if diff_mode:
        range_beg = 0
    else:
        range_beg = np.percentile(data_1d, 0)

    range_end = np.percentile(data_1d, 98)

    bounds = np.linspace(range_beg, range_end, 10)  # 10 bins
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")
    #  data_im = data_blockwise[:, :5000]
    #  data_im = data_blockwise[1155:6270, :]

    data_im = data_blockwise

    # Using the explicit API
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    im = ax.imshow(data_im, norm=norm, aspect="auto", cmap="plasma")
    ax.set_title(title)
    ax.set_xlabel("Block ID")
    ax.set_ylabel("Timestep")

    fig.tight_layout()

    cax = _fig_make_cax(fig)
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))

    PlotSaver.save(fig, trace_dir, None, plot_fname)


def plot_timegrid_rankwise(evt: str, data_rankwise: np.array, diff_mode=False):
    plot_fname, title = _evt_get_labels(evt, "rank", diff_mode=diff_mode)
    if diff_mode:
        data_im = _get_relative_timegrid(data_rankwise)
    else:
        data_im = data_rankwise

    data_1d = data_im.flatten()

    """ Could be 0th percentile, changed to avoid ts 8396-8418 issue """
    range_beg = np.percentile(data_1d, 0.5)
    range_end = np.percentile(data_1d, 98)

    bounds = np.linspace(range_beg, range_end, 10)  # 10 bins
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

    # Using the explicit API
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    im = ax.imshow(data_im, norm=norm, aspect="auto", cmap="plasma")
    ax.set_title(title)
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Timestep")

    fig.tight_layout()

    cax = _fig_make_cax(fig)
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))

    PlotSaver.save(fig, trace_dir, None, plot_fname)


def plot_timegrid_nodewise(evt: str, data_nodewise: np.array, diff_mode=False):
    plot_fname, title = _evt_get_labels(evt, "node", diff_mode=diff_mode)

    if diff_mode:
        data_im = _get_relative_timegrid(data_nodewise)
    else:
        data_im = data_nodewise

    data_1d = data_nodewise.flatten()

    """ Could be 0th percentile, changed to avoid ts 8396-8418 issue """
    range_beg = np.percentile(data_1d, 0.5)
    range_end = np.percentile(data_1d, 98)

    bounds = np.linspace(range_beg, range_end, 10)  # 10 bins
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")

    # Using the explicit API
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    im = ax.imshow(data_im, norm=norm, aspect="auto", cmap="plasma")
    ax.set_title(f"Nodewise times for evt {evt}")
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Timestep")

    fig.tight_layout()

    cax = _fig_make_cax(fig)
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))

    PlotSaver.save(fig, trace_dir, None, plot_fname)


def plot_timegrid_tuple(evt: str, evt_tuple: Tuple, nranks, nnodes):
    bw_1d, bw_2d, b1_1d_rel, bw_2d_rel = evt_tuple

    rw_2d, nw_2d = aggr_block_nparr(bw_2d, nranks=nranks, nnodes=nnodes)

    for diff_mode in [False, True]:
        plot_timegrid_blockwise(evt, bw_2d, bw_1d, diff_mode=diff_mode)
        plot_timegrid_rankwise(evt, rw_2d, diff_mode=diff_mode)
        plot_timegrid_nodewise(evt, nw_2d, diff_mode=diff_mode)


def plot_timegrid_all():
    evt0_tuple = _read_and_reg_evt("0", clip=5000)
    evt1_tuple = _read_and_reg_evt("1", clip=5000)
    evt0_1_tuple = _add_tuples([evt0_tuple, evt1_tuple])

    plot_timegrid_tuple("0", evt0_tuple, nranks=nranks, nnodes=nnodes)
    plot_timegrid_tuple("1", evt1_tuple, nranks=nranks, nnodes=nnodes)
    plot_timegrid_tuple("0+1", evt0_1_tuple, nranks=nranks, nnodes=nnodes)

    """
    Getting zero absolute values for nodewise data for timesteps:
    8396, 8397, 8398, 8399, 8400 ... 8418
    Presumably because of restart issues
    """


def get_evt_mat(evt_code):
    mat_path = f"{trace_dir}/evt{evt_code}.mat.pickle"
    mat = pickle.loads(open(mat_path, "rb").read())
    return mat


def filter_evt_mat(mat, evt_idx):
    all_rows = []
    for ts in mat:
        row = []
        for block in ts:
            if len(block) > evt_idx:
                row.append(block[evt_idx])
            else:
                row.append(0)
        all_rows.append(row)

    print(f"Rows: {len(all_rows)}")
    return all_rows


def plot_evt_instance_std():
    mat0 = get_evt_mat(0)
    mat1 = get_evt_mat(1)

    mat00 = filter_evt_mat(mat0, 0)
    std00 = list(map(np.std, mat00))

    mat01 = filter_evt_mat(mat0, 1)
    std01 = list(map(np.std, mat01))

    mat10 = filter_evt_mat(mat1, 0)
    std10 = list(map(np.std, mat10))

    mat11 = filter_evt_mat(mat1, 1)
    std11 = list(map(np.std, mat11))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    fname = "prof.evt.std.instancewise"

    # BABA
    ax.plot(range(len(std10)), std10, label="Evt1, 0 ($B_1$)")
    ax.plot(range(len(std00)), std00, label="Evt0, 0 ($A_1$)")
    ax.plot(range(len(std11)), std11, label="Evt1, 1 ($B_2$)")
    ax.plot(range(len(std01)), std01, label="Evt0, 1 ($A_2$)")

    ax.set_title("np.std for evts 0,1 (invocation-wise)")
    ax.set_xlabel("timestep")
    ax.set_ylabel("std-dev (ms)")

    ax.legend()

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f} ms".format(x / 1e3))
    ax.set_ylim([0, 3000])

    ax.yaxis.set_major_locator(MultipleLocator(400))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")

    fig.tight_layout()
    PlotSaver.save(fig, trace_dir, None, fname)


def plot_nblocks_from_assigndf():
    global trace_dir

    df_path = f"{trace_dir}/prof.aggrmore.evt3.csv"
    df = pd.read_csv(df_path)
    df["block_id"] = df["block_id"].apply(safe_ls)
    df["nblocks"] = df["block_id"].apply(len)

    data_y = df["nblocks"].to_numpy()
    data_x = df["sub_ts"].to_numpy()

    # in case sub_ts is negative
    data_x += -data_x[0]

    fig = plt.figure()
    ax = fig.subplots(1, 1)

    ax.plot(data_x, data_y)

    trace_name = trace_dir.split("/")[-1]
    ax.set_title(f"Block Count: Trace {trace_name}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Block Count")
    ax.set_ylim(bottom=0)

    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")

    fig.tight_layout()

    #  trace_name = os.path.basename(trace_dir)
    #  plot_fname = f"nblocks_{trace_name}"
    plot_fname = "nblocks"
    PlotSaver.save(fig, trace_dir, None, plot_fname)
    pass


"""
Analysis written on June 2, 2023.

Plot heatmaps and variances for each individual occurence of a kernel.
"""


def run_analyze_prof_times():
    # The matrix created via pickle keeps multiple invocations of a kernel # within a timestep separate. The numpy mat adds the two kernels.
    run_sep_by_evt_create_mat_eff(0)
    run_sep_by_evt_create_mat_eff(1)
    #  run_sep_by_evt_create_mat_eff(5)
    #  run_sep_by_evt_create_mat_eff(6)

    matplotlib.use("GTK3Agg")
    plt.ion()

    all_files = glob.glob(f"{trace_dir}/evt*.mat.pickle")
    all_files = sorted(all_files, key=lambda x: int(x.split(".")[-3][-1]))

    # 0, 1, 5, 6
    evt_labels = ["FillDerived", "CalcFluxes", "SNIAFeedback", "TabCooling"]

    all_files

    all_mats = list(map(lambda x: pickle.loads(open(x, "rb").read()), all_files))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    all_50 = list(map(lambda x: np.percentile(x, 50, axis=1), all_mats))
    all_90 = list(map(lambda x: np.percentile(x, 90, axis=1), all_mats))
    all_99 = list(map(lambda x: np.percentile(x, 99, axis=1), all_mats))

    data_x = np.arange(all_50[0].shape[0])
    ax.clear()

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))
    )
    ax.yaxis.grid(which="major", visible=True, color="#bbb")
    ax.xaxis.grid(which="major", visible=True, color="#bbb")
    ax.yaxis.grid(which="minor", visible=True, color="#ddd")

    ax.yaxis.set_major_locator(MultipleLocator(20 * 1e3))
    ax.yaxis.set_minor_locator(MultipleLocator(4 * 1e3))
    ax.xaxis.set_minor_locator(MultipleLocator(10 * 1e3))
    ax.xaxis.set_major_locator(MultipleLocator(20 * 1e3))

    ax.set_title("50, 90, and 99 percentile block kernel times")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (ms)")

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f}K".format(x / 1e3))
    )

    for idx, dy in enumerate(all_50):
        color = f"C{idx}"
        ls = "--"
        alpha = 0.4
        ax.plot(
            data_x,
            dy,
            color=color,
            linestyle=ls,
            zorder=2,
            alpha=alpha,
        )

    for idx, dy in enumerate(all_90):
        color = f"C{idx}"
        ls = "-."
        alpha = 0.7
        ax.plot(
            data_x,
            dy,
            color=color,
            linestyle=ls,
            zorder=2,
            alpha=alpha,
        )

    for idx, dy in enumerate(all_99):
        color = f"C{idx}"
        ls = "-"
        alpha = 1
        ax.plot(
            data_x,
            dy,
            color=color,
            linestyle=ls,
            zorder=2,
            label=evt_labels[idx],
            alpha=alpha,
        )

    ax.legend(ncol=4)
    fig.tight_layout()
    fname = "blktimes.50to99ptile"
    PlotSaver.save(fig, trace_dir, None, fname)

    return


def analyze(nranks):
    #  #  Create tracedir/prof.aggr.evtN.csv
    run_sep_by_evt(nranks)
    #  return

    #  # create aggrmore.X.csv
    blk_df = run_aggr_assns()
    ref_df = run_aggr_refs(nranks)

    #  # refinements.bin
    write_refinements()
    #  # assignments.bin
    write_assignments()

    # Validation suite
    validate_refinements(blk_df, ref_df)

    #  nblocks_0, prof_mat_0 = read_times(0)
    #  nblocks_1, prof_mat_1 = read_times(1)

    #  print("Validating nblocks from evt 0")
    #  validate_nblocks(blk_df, nblocks_0)

    #  print("Validating nblocks from evt 1")
    #  validate_nblocks(blk_df, nblocks_1)

    #  print("Computing stats on evt 0")
    #  compute_prof_mat_stats(prof_mat_0)

    #  print("Computing stats on evt 1")
    #  compute_prof_mat_stats(prof_mat_1)

    #  print("Computing stats on evt 0+1")
    #  compute_prof_mat_stats(prof_mat_0 + prof_mat_1)

    #  Only when a run had to be restarted
    #  run_join_two_traces()


def convert_pickled_mat_to_bin(trace_dir: str) -> None:
    n = 3
    m = 4
    mat = np.random.rand(n, m)
    mat_out = f"{trace_dir}/tmp.bin"

    skip_map = { 0: 0, 1: 3 }

    #  for evt_code in [0, 1, 5, 6]:
    for evt_code in [0, 1]:
        mat_in = f"{trace_dir}/evt{evt_code}.mat.pickle"
        mat_out = f"{trace_dir}/evt{evt_code}.mat.bin"
        #  mat_out = f"{trace_dir}/tmp.bin"

        print(f"Converting {mat_in} -> {mat_out}")

        mat = pickle.loads(open(mat_in, "rb").read())
        n = len(mat)
        ntoskip = skip_map[evt_code]
        print(f"\t - {n} timesteps found ({ntoskip} skipped) ...")

        with open(mat_out, "wb") as f:
            f.write(n.to_bytes(4, "little"))
            for ts in range(0, n):
                row = np.array(mat[ts], dtype=np.int32)
                m = len(row)
                tstoemit = ts + ntoskip
                f.write(tstoemit.to_bytes(4, "little"))
                f.write(m.to_bytes(4, "little"))
                f.write(row.tobytes())

def convert_npy_mat_to_bin(trace_dir: str) -> None:
    skip_map = { 0: 0, 1: 3}

    for evt_code in [0, 1]:
        mat_in = f"{trace_dir}/evt{evt_code}.mat.npy"
        mat_out = f"{trace_dir}/evt{evt_code}.mat.bin"
        #  mat_out = f"{trace_dir}/tmp.bin"

        print(f"Converting {mat_in} -> {mat_out}")
        mat = np.load(mat_in)
        mat = mat.astype(np.int32)
        print(mat)

        n = len(mat)
        #  n = 8
        ntoskip = skip_map[evt_code]
        print(f"\t - {n} timesteps found ({ntoskip} skipped) ...")

        with open(mat_out, "wb") as f:
            f.write(n.to_bytes(4, "little"))
            for ts in range(0, n):
                row = mat[ts]
                row = row[row != -2147483648]
                #  row = np.array(mat[ts], dtype=np.int32)
                m = len(row)
                #  print(f"Row len: {m}")
                tstoemit = ts + ntoskip
                #  print(row)
                #  return
                f.write(tstoemit.to_bytes(4, "little"))
                f.write(m.to_bytes(4, "little"))
                f.write(row.tobytes())


        return
    pass

def tmp():
    x = blk_df['block_id'].map(len)
    y = list(map(len, mat))
    len(x)
    len(y)
    sum(x) - sum(y) - sum(x[0:3])
    evt_code = 1


def plot():
    plot_init()
    #  evts = list(range(2))
    #  for evt in [0, 1]:
    #  for smooth in [0, 100]:
    #  print(f'Plotting aggr stats, evt: {evt}, smooth: {smooth}')
    #  plot_aggr_stats(evt, smooth)
    #  plot_rankhours(evts)

    # plots 0, 1, 0 + 1
    #  plot_timegrid_all()
    plot_nblocks_from_assigndf()


def run():
    nranks = 2048
    analyze(nranks)
    #  run_analyze_prof_times()
    #  run_analyze_compare_prof()
    #  plot()


if __name__ == "__main__":
    global trace_dir_fmt
    global trace_dir

    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    trace_dir = trace_dir_fmt.format("profile44")
    trace_dir = trace_dir_fmt.format("athenapk14")
    trace_dir = trace_dir_fmt.format("athenapk4")
    trace_dir = trace_dir_fmt.format("stochsg6")
    trace_dir = trace_dir_fmt.format("stochsg25")
    trace_dir = trace_dir_fmt.format("stochsg44")
    #  trace_dir = trace_dir_fmt.format("sparse1")
    trace_dir

    #  plot_init()

    run()
    #  convert_pickled_mat_to_bin(trace_dir)
    #  convert_npy_mat_to_bin(trace_dir)
    #  for trace in [38, 39, 41, 42]:
    #  trace_dir = trace_dir_fmt.format(trace)
    #  run()

    #  trace_dir = "/mnt/ltio/parthenon-topo/burgers2"
    #  run()
