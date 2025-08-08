import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from matplotlib import ticker
from common import PlotSaver, plot_init_big as plot_init


def plot_topobench_df(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    df["nranks"] = df["nranks"].astype(int)

    all_ranks = df["nranks"].unique()
    all_policies = ["baseline", "lpt"]
    all_nrounds = sorted(df["nrounds"].unique())

    colors = {
        512: "C0",
        1024: "C1",
        2048: "C2",
        4096: "C3",
    }

    linestyle = {
        "baseline": "-",
        "lpt": "--",
    }

    for nranks in all_ranks:
        for policy in all_policies:
            df_rank = df[(df["nranks"] == nranks) & (df["policy"] == policy)]
            data_x = df_rank["nrounds"]
            data_y = df_rank["t`ax_ms"]
            label = f"{nranks} ranks ({policy})"

            color = colors[nranks]
            ls = linestyle[policy]

            ax.plot(data_x, data_y, label=label, color=color, linestyle=ls)

    ax.set_xlabel("Number of rounds")
    ax.set_ylabel("Time (ms)")

    ax.set_xticks(all_nrounds)

    ax.xaxis.yet_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.grid(which="minor", color="#ddd")
    ax.grid(which="major", color="#bbb")

    ax.legend()

    fig.tight_layout()

    plot_fname = "test"
    PlotSaver.save(fig, "", None, plot_fname)


def process_trace(trace_path: str) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    df[["time_max_ms", "num_obs", "meshgen_method"]]
    df_rel = df[["meshgen_method", "num_obs", "time_max_ms"]].copy()
    df_rel.columns = ["trace", "nrounds", "tmax_ms"]

    df_rel["policy"] = df_rel["trace"].str.split(".").map(lambda x: x[3])
    df_rel["nbrtype"] = df_rel["trace"].str.split(".").map(lambda x: x[4])
    df_rel["nranks"] = (
        df_rel["trace"].str.split(".").map(lambda x: re.findall(r"\d+$", x[0])[0])
    )

    # drop column trace
    df_rel = df_rel.drop(columns=["trace"])

    return df_rel


def compute_fixed_and_marginal_times(df: pd.DataFrame) -> dict:
    all_nranks = df["nranks"].unique()

    rank_times: dict[int, list[float]] = {}

    for nranks in all_nranks:
        df_nranks = df[df["nranks"] == nranks]
        nrounds = df_nranks["nrounds"].to_list()
        tmax = df_nranks["tmax_ms"].to_list()

        assert nrounds[0] == 1

        t0 = tmax[0]
        rank_times[nranks] = [t0]

        other_nrounds = np.array(nrounds[1:])
        other_times = np.array(tmax[1:])

        other_times = other_times - t0
        other_nrounds = other_nrounds - nrounds[0]

        marginal_times = other_times / other_nrounds
        rank_times[nranks].extend(marginal_times)

    return rank_times


def plot_marginal_times_bar(
    rank_times: dict[int, list[float]], title_suffix: str, plot_fname: str
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    all_nranks = sorted(list(rank_times.keys()))
    bar_width = 1 / (len(all_nranks) + 1)

    nrounds = max(len(v) for v in rank_times.values())
    all_nrounds = list(range(nrounds))
    all_nrounds_labels = [f"Next {i}\n(avg)" for i in all_nrounds]
    all_nrounds_labels[0] = "First"

    for i, nranks in enumerate(all_nranks):
        times = rank_times[nranks]
        x = np.arange(len(times))

        ax.bar(
            x + i * bar_width, times, width=bar_width, label=f"{nranks} ranks", zorder=2
        )

    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Marginal Time/Round {title_suffix}")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} ms"))

    ax.set_xticks(all_nrounds)
    ax.set_xticklabels(all_nrounds_labels)

    ax.set_ylim([-5, 100])

    ax.legend()

    fig.tight_layout()
    PlotSaver.save(fig, "", None, plot_fname)


def average_out_single_df(df: pd.DataFrame) -> pd.DataFrame:
    df["nranks"] = df["nranks"].astype(int)
    df_avg = df.groupby(["policy", "nbrtype", "nranks", "nrounds"]).mean().reset_index()
    df_avg.sort_values(by=["policy", "nbrtype", "nranks", "nrounds"], inplace=True)
    return df_avg


def plot_marginal_times(trace_df: pd.DataFrame, trace_id: str) -> None:
    df_baseline = trace_df[trace_df["policy"] == "baseline"].to_frame()
    df_baseline = average_out_single_df(df_baseline)
    costs_baseline = compute_fixed_and_marginal_times(df_baseline)

    title_suffix = f"({trace_id}, Baseline)"
    trace_id_fname = trace_id.lower().replace(" ", ".").replace(",", "")
    plot_fname = f"topobench.marginal_times.{trace_id_fname}.baseline"
    plot_marginal_times_bar(costs_baseline, title_suffix, plot_fname)

    df_lpt = trace_df[trace_df["policy"] == "lpt"].to_frame()
    df_lpt = average_out_single_df(df_lpt)
    costs_lpt = compute_fixed_and_marginal_times(df_lpt)

    title_suffix = f"({trace_id}, LPT)"
    trace_id_fname = trace_id.lower().replace(" ", ".").replace(",", "")
    plot_fname = f"topobench.marginal_times.{trace_id_fname}.lpt"
    plot_marginal_times_bar(costs_lpt, title_suffix, plot_fname)


def run_plot_marginal_times():
    df_root = "/Users/schwifty/Repos/amr-data/20240702"

    df_path = f"{df_root}/20240701_bench_log_sorted.csv"
    trace_df = process_trace(df_path)

    df_uniform = trace_df[trace_df["nbrtype"] == "uniform"].to_frame()
    plot_marginal_times(df_uniform, "Sorted, Uniform")

    df_regular = trace_df[trace_df["nbrtype"] == "reg"].to_frame()
    plot_marginal_times(df_regular, "Sorted, Regular")

    df_path = f"{df_root}/20240702_bench_log_randomized.csv"
    trace_df = process_trace(df_path)

    df_uniform = trace_df[trace_df["nbrtype"] == "uniform"].to_frame()
    plot_marginal_times(df_uniform, "Randomized, Uniform")

    df_regular = trace_df[trace_df["nbrtype"] == "reg"].to_frame()
    plot_marginal_times(df_regular, "Randomized, Regular")


def run():
    run_plot_marginal_times()


if __name__ == "__main__":
    plot_init()
    cmap = plt.colormaps["Dark2"]
    colors = [cmap(i) for i in range(cmap.N)]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    run()
