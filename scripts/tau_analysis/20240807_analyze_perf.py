import glob
import re
import multiprocessing
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from common import PlotSaver


def read_perf(perf_file: str) -> tuple[int, int, float]:
    mobj = re.findall(r"perf.data\.(\d+)\.txt", perf_file)
    rank = int(mobj[0])

    with open(perf_file, "r") as f:
        lines = [float(l.strip()) for l in f.readlines()]
        return (rank, len(lines), sum(lines))


def read_all_perf(perf_dir: str) -> pd.DataFrame:
    glob_patt = f"{perf_dir}/prof/perf.data.*.csv"
    all_files = glob.glob(glob_patt)
    all_files

    with multiprocessing.Pool(16) as pool:
        all_data = pool.map(pd.read_csv, all_files)

    data_df = pd.concat(all_data).sort_values("rank")
    data_df.columns = ["func", "count", "time", "rank"]

    return data_df


def plot_counts(df: pd.DataFrame, func: str, plot_fname: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    df_func = df[df["func"] == func].sort_values("rank")
    dx = df_func["rank"]
    dy_a = df_func["count_a"]
    dy_b = df_func["count_b"]

    ax.plot(dx, dy_a, label=f"{func}_cdp")
    ax.plot(dx, dy_b, label=f"{func}_h25")
    ax.plot(dx, dy_b - dy_a, label=f"{func}_diff")

    ax.set_title(f"Counts for {func}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Counts")

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#eee")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.legend()

    fig.tight_layout()

    plot_fname = f"{plot_fname}_{func}_counts".replace(":", "")
    PlotSaver.save(fig, "", None, plot_fname)


def plot_times(df: pd.DataFrame, func: str, plot_fname: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    df_func = df[df["func"] == func].sort_values("rank")
    dx = df_func["rank"]
    dy_a = df_func["time_a"]
    dy_b = df_func["time_b"]

    ax.plot(dx, dy_a, label=f"{func}_cdp")
    ax.plot(dx, dy_b, label=f"{func}_h25")
    ax.plot(dx, dy_b - dy_a, label=f"{func}_diff")

    ax.set_title(f"Times for {func}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Counts")

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#eee")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.legend()

    fig.tight_layout()

    plot_fname = f"{plot_fname}_{func}_times".replace(":", "")
    PlotSaver.save(fig, "", None, plot_fname)


def plot_times_diff_of_diff(df: pd.DataFrame, plot_fname: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    df_isend = df[df["func"] == "mpi:isend"].sort_values("rank")
    diff_isend = df_isend["time_b"] - df_isend["time_a"]

    df_conn = df[df["func"] == "ips:conn"].sort_values("rank")
    diff_conn = df_conn["time_b"] - df_conn["time_a"]

    diff_diff = diff_isend.to_numpy() - diff_conn.to_numpy()
    dx = df_isend["rank"]

    ax.plot(dx, diff_diff)

    ax.set_title(f"Residual isend time")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Counts")

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#eee")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.legend()

    fig.tight_layout()

    plot_fname = f"{plot_fname}_diffofdiff"
    PlotSaver.save(fig, "", None, plot_fname)


def run():
    perf_run = "/mnt/ltio/parthenon-topo/ub22perfdbg/blastw4096.10.perfdbg"
    perf_dir_cdp = f"{perf_run}.cdpc512par8"
    perf_dir_h25 = f"{perf_run}.hybrid25"
    perf_dir_h10 = f"{perf_run}.hybrid10"

    df_cdp = read_all_perf(perf_dir_cdp)
    df_h25 = read_all_perf(perf_dir_h25)
    df_h10 = read_all_perf(perf_dir_h10)

    df_merged = pd.merge(df_cdp, df_h25, on=["func", "rank"], suffixes=("_a", "_b"))
    df_merged

    trace_name = os.path.basename(perf_run)

    plot_counts(df_merged, "mpi:isend", trace_name)
    plot_counts(df_merged, "ips:conn", trace_name)
    plot_times(df_merged, "mpi:isend", trace_name)
    plot_times(df_merged, "ips:conn", trace_name)

    plot_times_diff_of_diff(df_merged, trace_name)

    df_merged = pd.merge(df_cdp, df_h10, on=["func", "rank"], suffixes=("_a", "_b"))
    trace_name = f"{trace_name}.h10"
    plot_times(df_merged, "mpi:isend", trace_name)


if __name__ == "__main__":
    run()
