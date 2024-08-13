import glob
import multiprocessing
import os
import re
import struct

import pandas as pd

from typing import cast


def read_msgs(fpath):
    mobj = re.search(r"msgs.(\d+).bin$", fpath)
    assert mobj is not None

    rank = int(mobj.groups(0)[0])
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
    trace_dir: str, rank_beg: int, rank_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_fmt = "{0}/trace/msgs/msgs.{1}.bin"
    all_bins = [path_fmt.format(trace_dir, r) for r in range(rank_beg, rank_end + 1)]
    print(f"Reading {all_bins} msgs.bin")

    for p in all_bins:
        assert os.path.exists(p)

    with multiprocessing.Pool(16) as p:
        all_dfs = p.map(read_msgs, all_bins)

    chan_aggr = pd.concat(map(lambda x: x[0], all_dfs))
    chan_df: pd.DataFrame = cast(pd.DataFrame, chan_aggr)
    send_aggr = pd.concat(map(lambda x: x[1], all_dfs))
    send_df: pd.DataFrame = cast(pd.DataFrame, send_aggr)

    chan_df.sort_values(["rank", "ts"], inplace=True)
    send_df.sort_values(["rank", "ts"], inplace=True)

    return chan_df, send_df


def aggr_msgs_all(trace_dir: str):
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

    cols_to_keep = ["ts", "blk_id", "blk_rank", "nbr_id", "nbr_rank", "msgsz", "isflx"]
    joined_df = joined_df[cols_to_keep]

    return joined_df


def run_aggr_msgs_inner(trace_dir: str):
    chan_df, send_df = aggr_msgs_all(trace_dir)
    joined_df = join_msgs(send_df, chan_df)

    joined_out = f"{trace_dir}/trace/msgs.aggr.joined.csv"

    # print(f"Writing to {joined_out}")
    # joined_df.to_csv(joined_out, index=None)

    trace_name = os.path.basename(trace_dir)
    trace_root = os.path.dirname(trace_dir)
    joined_df_path = f"{trace_root}/{trace_name}.csv"
    print(f"Writing to {joined_df_path}")
    joined_df.to_csv(joined_df_path, index=None)


"""
analyze_msgs: check out the min/median/max no. of msgs/rank
"""


def analyze_msgs(trace_dir: str):
    joined_df = pd.read_csv(f"{trace_dir}.csv")
    blk_map = joined_df[["ts", "blk_id", "blk_rank"]].copy()
    blk_map.drop_duplicates(inplace=True)

    blkcnt_map = blk_map.groupby(["ts", "blk_rank"], as_index=False).agg(
        {"blk_id": "count"}
    )
    blkcnt_map.columns = ["ts", "blk_rank", "blk_cnt"]

    stat_df = blkcnt_map.groupby(["ts"], as_index=False).agg(
        {
            "blk_cnt": [
                "min",
                "max",
                "median",
            ]
        }
    )

    print(f"Stats for {trace_dir}: \n", stat_df)


"""
make_uniform_msgtrace: extract timestep 2 from the main msgtrace
create two variants of it - reg (which is untouched), and 
uniform (which only has one block per rank)
"""


def make_uniform_msgtrace(trace_dir: str):
    joined_df = pd.read_csv(f"{trace_dir}.csv")
    msg_df = joined_df[joined_df["ts"] == 2].copy()
    msg_df.drop(columns=["ts"], inplace=True)
    msg_df.reset_index(inplace=True, drop=True)
    msg_df["msg_id"] = msg_df.index

    firstblk_map = msg_df.groupby(["blk_rank"], as_index=False).agg({"blk_id": "min"})

    # join firstblk_map with msg_df, on blk_id and blk_id
    send_df = firstblk_map.merge(
        msg_df, how="inner", on=["blk_id", "blk_id"], suffixes=("_x", None)
    )
    send_df.drop(columns=["blk_rank_x"], inplace=True)

    recv_df = firstblk_map.merge(
        msg_df,
        how="inner",
        left_on=["blk_id"],
        right_on=["nbr_id"],
        suffixes=("_x", None),
    )

    # drop columns blk_rank_x, blk_id_x on recv_df
    # rename blk_id_y, blk_rank_y to blk_id, blk_rank
    recv_df.drop(columns=["blk_rank_x", "blk_id_x"], inplace=True)

    uniform_msgdf = pd.concat([send_df, recv_df]).drop_duplicates(subset="msg_id")

    len_msgdf = len(msg_df)
    len_unifmsgdf = len(uniform_msgdf)

    print(f"msg_df_len: {len_msgdf}, uniform_msgdf_len: {len_unifmsgdf}")
    print(f"ratio: {len_unifmsgdf/len_msgdf:.2f}")

    trace_name = os.path.basename(trace_dir)
    trace_root = os.path.dirname(trace_dir)

    msgdf_fname = f"{trace_name}.reg.csv"
    uniform_msgdf_fname = f"{trace_name}.uniform.csv"

    msgdf_fpath = f"{trace_root}/ts2/{msgdf_fname}"
    uniform_msgdf_fpath = f"{trace_root}/ts2/{uniform_msgdf_fname}"

    print(f"Writing to {msgdf_fpath}")
    msg_df.to_csv(msgdf_fpath, index=None)

    print(f"Writing to {uniform_msgdf_fpath}")
    uniform_msgdf.to_csv(uniform_msgdf_fpath, index=None)


def run_aggr_msgs():
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/msgtrace/{}"

    ranks = [512, 1024, 2048, 4096]
    policies = ["baseline", "lpt"]

    for r in ranks:
        for p in policies:
            trace_name = "blastw{}.msgtrace.01.{}".format(r, p)
            trace_dir = trace_dir_fmt.format(trace_name)
            print(f"Processing trace: {trace_dir}")
            # run_aggr_msgs_inner(trace_dir)
            # analyze_msgs(trace_dir)
            make_uniform_msgtrace(trace_dir)


def run():
    run_aggr_msgs()


if __name__ == "__main__":
    run()
