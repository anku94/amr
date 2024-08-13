import io
import os
import re
import time

import pandas as pd


def collapse_df(df: pd.DataFrame, coll_id: int) -> pd.DataFrame:
    df["group"] = (df["id"] != coll_id).cumsum()
    df_collapsed = df.groupby(["group", "id"]).agg({"time": "sum"}).reset_index()
    df_collapsed.drop(columns="group", inplace=True)

    return df_collapsed


def parse_rank_inner(df_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tswise_lines = open(df_path).readlines()
    # 500 is a good estimate for max event count
    last_section = tswise_lines[-500:]

    name_regex = re.compile(r"(\d+)\ ([a-zA-Z].+)\n")
    names = [re.match(name_regex, l) for l in last_section]
    names = [n.groups() for n in names if n]

    names = list(zip(*names))
    ids = [int(i) for i in names[0]]
    names = names[1]
    assert len(ids) == max(ids) + 1
    name_df = pd.DataFrame({"id": ids, "name": names})
    name_df.set_index("id", inplace=True)
    name_df.sort_index(inplace=True)

    data = "".join(tswise_lines[: -len(names)])
    data_df = pd.read_csv(
        io.StringIO(data),
        sep=" ",
        header=None,
        names=["id", "time"],
        dtype=int,
        on_bad_lines="warn",
    )

    return (name_df, data_df)


def save_df_with_retry(df: pd.DataFrame, fpath: str):
    for attempt in range(100):
        try:
            df.to_feather(fpath)
            break
        except Exception as e:
            print(f"Error saving {fpath}: {e}, attempt {attempt}")
            time.sleep(3)


def parse_rank(trace_path: str, rank: int):
    tswise_fpath = f"{trace_path}/trace/amrmon_tswise_{rank}.txt"

    name_df, data_df = parse_rank_inner(tswise_fpath)
    most_freq = data_df["id"].mode().values[0]
    simple_df = collapse_df(data_df, most_freq)

    get_id = lambda s: name_df[name_df["name"] == s].index[0]

    # filter Driver_Main
    main_idx = get_id("Driver_Main")
    simple_df = simple_df[simple_df["id"] != main_idx]

    # assign timestep
    mo_idx = get_id("MakeOutputs")
    simple_df["timestep"] = (simple_df["id"] == mo_idx).cumsum()
    data_df["timestep"] = (data_df["id"] == mo_idx).cumsum()

    # identify lb ts
    lbs1_idx = get_id("Step1: Construct new list")
    lb_ts = simple_df[simple_df["id"] == lbs1_idx]["timestep"]
    lb_df = simple_df[simple_df["timestep"].isin(lb_ts)].copy()

    assert type(lb_df) == pd.DataFrame

    lb_out = f"{trace_path}/trace/lb_aggr_{rank}.feather"
    names_out = f"{trace_path}/trace/names_{rank}.feather"
    # data_out = f"{trace_path}/trace/data_{rank}.feather"

    save_df_with_retry(lb_df, lb_out)
    save_df_with_retry(name_df, names_out)


def run_mpich():
    prof_dir = os.environ["AMRMON_OUTPUT_DIR"]
    rank = int(os.environ["PMI_RANK"])
    run_dir = os.path.dirname(prof_dir)

    print(f"Profiling rank {rank} in {prof_dir}")
    parse_rank(run_dir, rank)


if __name__ == "__main__":
    run_mpich()
