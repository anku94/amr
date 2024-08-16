import glob
import multiprocessing
import os
import re
import sys
import time

import pandas as pd

NAME_CACHE = {}


def save_df_with_retry(df: pd.DataFrame, fpath: str):
    for attempt in range(100):
        try:
            df.to_feather(fpath)
            break
        except Exception as e:
            print(f"Error saving {fpath}: {e}, attempt {attempt}")
            time.sleep(3)


def get_rank(fpath: str) -> int:
    fname = os.path.basename(fpath)
    mobj = re.match(r".*_(\d+)\.feather", fname)
    assert mobj is not None
    return int(mobj.groups()[0])


def func2str(func: str) -> str:
    return func.replace(":", "").replace(" ", "").lower()


def func2fpath(trace_fpath: str, func: str) -> str:
    return f"{trace_fpath}/prof/{func2str(func)}.feather"


def read_all_names(trace_path: str) -> pd.DataFrame:
    names_out = f"{trace_path}/prof/names_aggr.feather"

    if trace_path in NAME_CACHE:
        return NAME_CACHE[trace_path]

    if os.path.exists(names_out):
        print(f"Reading {names_out}")
        all_names = pd.read_feather(names_out)
        NAME_CACHE[trace_path] = all_names
        return all_names

    names_files = glob.glob(f"{trace_path}/trace/names_*.feather")
    names_files = sorted(names_files, key=get_rank)

    with multiprocessing.Pool(16) as pool:
        names = pool.map(pd.read_feather, names_files)

    for f, df in zip(names_files, names):
        df["rank"] = get_rank(f)

    all_names = pd.concat(names)
    all_names["id"] = all_names.index
    all_names.rename(columns={"id": "id_old"}, inplace=True)

    unique_id = all_names[["name"]].drop_duplicates().reset_index(drop=True)
    unique_id["id_new"] = unique_id.index

    all_names = all_names.merge(unique_id, on="name", how="left")

    NAME_CACHE[trace_path] = all_names
    save_df_with_retry(all_names, names_out)

    return all_names


def read_data_single(data_file: str) -> pd.DataFrame:
    print(f"Reading {data_file}")

    df = pd.read_feather(data_file)
    df = (
        df[["timestep", "id", "time"]]
        .groupby(["timestep", "id"])
        .agg({"time": "sum"})
        .reset_index()
    )
    df["rank"] = get_rank(data_file)

    return df


def read_data(trace_path: str):
    data_files = glob.glob(f"{trace_path}/trace/lb_aggr_*.feather")
    data_files = sorted(data_files, key=get_rank)

    with multiprocessing.Pool(16) as pool:
        data = pool.map(read_data_single, data_files)

    all_data = pd.concat(data)

    return all_data


"""
all _ff functions operate on a specific func
"""


def get_names_ff(func: str, all_names: pd.DataFrame):
    return all_names[all_names["name"] == func]["id_old"].to_list()


def read_data_single_ff(data_file: str, func_id: int) -> pd.DataFrame:
    print(f"Reading {data_file}")

    df = pd.read_feather(data_file)
    df = (
        df[["timestep", "id", "time"]]
        .groupby(["timestep", "id"])
        .agg({"time": "sum"})
        .reset_index()
    )
    df["rank"] = get_rank(data_file)
    df = df[df["id"] == func_id]
    assert type(df) == pd.DataFrame

    return df


def read_data_serial_ff(data_arg_tuples: list[tuple[str, int]]) -> pd.DataFrame:
    all_dfs = list(map(lambda x: read_data_single_ff(*x), data_arg_tuples))
    df = pd.concat(all_dfs)
    return df


def chunk_list(ls: list, nchunks: int) -> list[list]:
    chunksz = (len(ls) + nchunks - 1) // nchunks
    return [ls[i : i + chunksz] for i in range(0, len(ls), chunksz)]


def gen_amrmon_aggr_ff(trace_path: str, func: str) -> None:
    all_names = read_all_names(trace_path)
    if func not in all_names["name"]:
        print(f"ALERT!! {func} does not exist in names!!")
        sys.exit(-1)

    func_ids = get_names_ff(func, all_names)

    data_files = glob.glob(f"{trace_path}/trace/lb_aggr_*.feather")
    data_files = sorted(data_files, key=get_rank)

    func_data_pairs = list(zip(data_files, func_ids))
    fdchunks = chunk_list(func_data_pairs, 16)

    with multiprocessing.Pool(16) as pool:
        # data = pool.starmap(read_data_single_ff, func_data_pairs)
        data = pool.map(read_data_serial_ff, fdchunks)

    all_data = pd.concat(data)
    all_data = all_data[["timestep", "rank", "time"]]
    func_mat = all_data.pivot(index="timestep", columns="rank", values="time")

    func_fpath = f"{trace_path}/prof/{func2str(func)}.feather"
    print(f"Writing to {func_fpath}")
    func_mat.to_feather(func_fpath)


def gen_amrmon_aggr(trace_path: str):
    all_names = read_all_names(trace_path)
    all_data = read_data(trace_path)

    slim_names = all_names[["rank", "id_old", "id_new"]].copy()

    merged_data = all_data.merge(
        slim_names, left_on=["rank", "id"], right_on=["rank", "id_old"], how="left"
    )
    merged_data.drop(columns=["id", "id_old"], inplace=True)
    merged_data.rename(columns={"id_new": "id"}, inplace=True)

    lb_aggr = merged_data[["timestep", "id", "rank", "time"]].sort_values(
        ["timestep", "id", "rank"]
    )

    lb_aggr = (
        lb_aggr.groupby(["timestep", "id", "rank"]).agg({"time": "sum"}).reset_index()
    )

    lb_names = all_names[["name", "id_new"]].drop_duplicates().reset_index(drop=True)
    assert isinstance(lb_names, pd.DataFrame)
    lb_names.rename(columns={"id_new": "id"}, inplace=True)

    lb_aggr_out = f"{trace_path}/lb_aggr.feather"
    print(f"Writing to {lb_aggr_out}")
    lb_aggr.to_feather(lb_aggr_out)

    lb_names_out = f"{trace_path}/lb_names.feather"
    print(f"Writing to {lb_names_out}")
    lb_names.to_feather(lb_names_out)
