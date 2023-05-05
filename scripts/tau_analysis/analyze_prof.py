import glob
import multiprocessing
import numpy as np
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import re
import subprocess
import struct
import sys
import time

import ray
import traceback

from common import plot_init, PlotSaver, prof_evt_map
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import List, Tuple
from pathlib import Path
from task import Task
from trace_reader import TraceReader, TraceOps


class ProfOutputReader(Task):
    def __init__(self, trace_dir: str, evt: int):
        super().__init__(trace_dir)
        self.evt = evt

    @staticmethod
    def worker(args):
        trace_dir = args["trace_dir"]
        r = args["rank"]
        evt = args["evt"]

        print(f"Running ProfReader for Rank: {r}, Evt: {evt}")
        trace_reader = TraceReader(trace_dir)
        return trace_reader.read_rank_prof(r, evt)
        pass

    def gen_worker_fn_args(self):
        args = super().gen_worker_fn_args()
        args["evt"] = self.evt
        return args

    #  def gen_worker_fn_args_rank(self, rank):
    #  args = self.gen_worker_fn_args()
    #  args["rank"] = rank
    #  return args


def get_prof_path(evt: int) -> str:
    global trace_dir
    # replace merged with agg if single self-contained run
    ppath = f"{trace_dir}/prof.merged.evt{evt}.csv"
    return ppath


def run_sep_by_evt_util(evt):
    global trace_dir
    tr = TraceReader(trace_dir)
    reader = ProfOutputReader(trace_dir, evt)
    all_dfs = reader.run_rankwise(0, 512)
    aggr_df = pd.concat(all_dfs)

    aggr_df.sort_values(["ts", "sub_ts", "rank", "block_id"], inplace=True)

    col_names = ["time_us", "time_us", "refine_flag", "block_idx"]
    cols_new = list(aggr_df.columns)[:-1]
    cols_new.append(col_names[evt])
    aggr_df.columns = cols_new

    evt_df_path = get_prof_path(evt)
    print(f"Writing evt {evt} to {evt_df_path}...")
    print(aggr_df)
    aggr_df.to_csv(evt_df_path, index=None)


def run_sep_by_evt():
    evts = [0, 1]
    for evt in evts:
        run_sep_by_evt_util(evt)


def group_by_ts(ref_df):
    boundaries = ref_df["block_id"] == -1

    all_refs = []

    prev_bidx = -1
    for bidx in ref_df[boundaries].index:
        df_cur = ref_df.loc[prev_bidx + 1 : bidx - 1]
        uniq_ts = df_cur["ts"].unique()
        assert len(uniq_ts) <= 1
        if len(uniq_ts) == 0:
            prev_bidx = bidx
            continue

        cur_ref_tuple = (
            uniq_ts[0],
            list(df_cur["block_id"]),
            list(df_cur["refine_flag"]),
        )

        all_refs.append(cur_ref_tuple)

        prev_bidx = bidx

    return all_refs


def run_aggr_evt3():
    evt_code = 3
    global trace_dir
    df_path = f"{trace_dir}/prof.aggr.evt{evt_code}.csv"
    df_out_path = f"{trace_dir}/prof.aggrmore.evt{evt_code}.csv"

    df = pd.read_csv(df_path)

    nranks = df["rank"].max() + 1

    all_dfs = []

    for rank in range(nranks):
        ref_df = df[df["rank"] == rank].copy()
        ref_df["sub_ts"] = ref_df["block_idx"].eq(0).cumsum() - 1
        all_dfs.append(ref_df)

    concat_df = pd.concat(all_dfs)

    aggr_df = concat_df.groupby("sub_ts", as_index=False).agg(
        {"ts": "min", "rank": list, "block_id": list}
    )

    aggr_df.to_csv(df_out_path, index=None)

    return aggr_df


def run_aggr_evt4():
    evt_code = 2
    global trace_dir
    df_out_path = f"{trace_dir}/prof.aggrmore.evt{evt_code}.csv"
    reader = ProfOutputReader(trace_dir, evt_code)
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

    #  merged_df = blk_df.merge(aggr_df, how="left", on="sub_ts").fillna(
    #  0, downcast="infer"
    #  )

    #  tmp_df = merged_df[["nblocks_before", "nref", "nderef", "nblocks_after"]].copy()
    #  tmp_df["nblocks_after_computed"] = (
    #  tmp_df["nblocks_before"] + 7 * tmp_df["nref"] - 7 * tmp_df["nderef"] / 8
    #  )

    #  try:
    #  assert (tmp_df["nblocks_after_computed"] == tmp_df["nblocks_after"]).all()
    #  print("Refinements validated. All block counts as expected!!")
    #  except AssertionError as e:
    #  mismatch_df = tmp_df[
    #  tmp_df["nblocks_after_computed"] != tmp_df["nblocks_after"]
    #  ]
    #  print(mismatch_df)

    #  print(deref_df[["ts", "nblocks", "nref", "nderef"]])

    #  for index, data in aggr_df.iterrows():
    #  print(index)
    #  print(data)
    #  break

    #  print(df)
    #  print(aggr_df)
    #  print(aggr_df[ "ts", "block_id", "refine_flag" ])


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
    tmp_df["nblocks_next_ts_computed"] = (
        tmp_df["nblocks_cur_ts"] + 7 * tmp_df["nref"] - 7 * tmp_df["nderef"] / 8
    )

    try:
        assert (tmp_df["nblocks_next_ts_computed"] == tmp_df["nblocks_next_ts"]).all()
        print("Refinements validated. All block counts as expected!!")
    except AssertionError as e:
        mismatch_df = tmp_df[
            tmp_df["nblocks_after_computed"] != tmp_df["nblocks_after"]
        ]
        print(mismatch_df)
        return


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
            write_int(f, len(ranks))

            for i in ranks:
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


def _aggr_nparr_by_rank(np_arr, nranks=512):
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


def aggr_block_nparr(np_arr, nranks=512, nnodes=32) -> Tuple[np.array, np.array]:
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
    df_path = get_prof_path(evt)
    print(f"Reading dataframe: {df_path}")

    df = pd.read_csv(df_path)
    df_agg = df.groupby("ts", as_index=False).agg({"time_us": list})

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


def plot_timegrid_blockwise(
    evt: str, data_blockwise: np.array, data_1d: np.array, diff_mode=False
):
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
    data_im = data_blockwise[:, :5000]

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


def plot_timegrid_tuple(evt: str, evt_tuple: Tuple, nranks=512, nnodes=32):
    bw_1d, bw_2d, b1_1d_rel, bw_2d_rel = evt_tuple

    rw_2d, nw_2d = aggr_block_nparr(bw_2d, nranks=nranks, nnodes=nnodes)

    for diff_mode in [False, True]:
        plot_timegrid_blockwise(evt, bw_2d, bw_1d, diff_mode=diff_mode)
        plot_timegrid_rankwise(evt, rw_2d, diff_mode=diff_mode)
        plot_timegrid_nodewise(evt, nw_2d, diff_mode=diff_mode)


def plot_timegrid_all():
    nranks = 512
    nnodes = 32

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


def analyze():
    #  Create tracedir/prof.aggr.evtN.csv
    run_sep_by_evt()
    #  run_group_by_ts()
    #  blk_df = run_aggr_evt3()
    #  ref_df = run_aggr_evt4()
    #  validate_refinements(blk_df, ref_df)
    #  write_refinements()
    #  write_assignments()
    #  Only when a run had to be restarted
    #  run_join_two_traces()


def plot():
    plot_init()
    #  evts = list(range(2))
    #  for evt in [0, 1]:
    #  for smooth in [0, 100]:
    #  print(f'Plotting aggr stats, evt: {evt}, smooth: {smooth}')
    #  plot_aggr_stats(evt, smooth)
    #  plot_rankhours(evts)

    # plots 0, 1, 0 + 1
    plot_timegrid_all()


def run():
    analyze()
    #  plot()


if __name__ == "__main__":
    global trace_dir
    trace_dir = "/mnt/ltio/parthenon-topo/profile22"
    run()
