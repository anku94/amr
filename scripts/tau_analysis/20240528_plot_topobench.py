import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import ticker

from common import plot_init_big as plot_init, PlotSaver


def linear_regression_closed_form(X, Y):
    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(Y)
    sum_xy = np.sum(X * Y)
    sum_x2 = np.sum(X * X)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    c = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)

    return m, c


def get_nranks(trace_path: str) -> int:
    # assuming fmt = blastw$nranks.something.$policy.csv
    trace_name = os.path.basename(trace_path)
    trace_first = trace_name.split(".")[0]

    matches = re.findall(r"\d+", trace_first)
    assert len(matches) == 1

    return int(matches[0])


def get_policy(trace_path: str) -> str:
    # assuming fmt = blastw$nranks.something.$policy.csv
    trace_name = os.path.basename(trace_path)
    policy = trace_name.split(".")[-2]
    return policy


def prep_data(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)

    df = df.rename(columns={"num_obs": "nrounds"})
    df["nranks"] = df["topology"].apply(get_nranks)
    df["policy"] = df["topology"].apply(get_policy)

    coi = ["nranks", "policy", "nrounds", "time_avg_ms", "time_max_ms"]
    df2 = df[coi].copy()

    aggr_df = (
        df2.groupby(["nranks", "policy", "nrounds"])
        .agg({"time_avg_ms": "mean", "time_max_ms": "mean"})
        .reset_index()
    )

    aggr_df.sort_values(by=["nranks", "policy", "nrounds"], inplace=True)
    aggr_df = aggr_df.astype({"nranks": int, "nrounds": int})

    return aggr_df


def plot_data(aggr_df: pd.DataFrame, plot_fname: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    cmap = plt.get_cmap("tab20")

    all_nranks = aggr_df["nranks"].unique()
    all_policies = aggr_df["policy"].unique()

    cmap_idx = 0
    for nridx, nranks in enumerate(all_nranks):
        for pidx, policy in enumerate(all_policies):
            df = aggr_df[(aggr_df["nranks"] == nranks) & (aggr_df["policy"] == policy)]
            dx = df["nrounds"].astype(int)
            dy_avg = df["time_avg_ms"]
            dy_max = df["time_max_ms"]

            # color = cmap(cmap_idx)
            # cmap_idx += 1
            #
            # ax.plot(
            #     dx,
            #     dy_avg,
            #     label=f"{nranks}/{policy} (avg)",
            #     color=color,
            #     linestyle="--",
            # )
            cmap_idx = nridx * 2 + (1 - pidx)
            color = cmap(cmap_idx)

            ax.plot(dx, dy_max, label=f"{nranks}/{policy} (max)", color=color)

    ax.set_xlabel("Number of rounds")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Topobench: BC Time vs Scale/Policy/Rounds")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} ms"))
    ax.set_ylim(bottom=0)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")

    # legend - 3 cols
    ax.legend(ncol=3)
    # ax.legend()
    fig.tight_layout()

    plot_fname = f"{plot_fname}_maxonly"
    PlotSaver.save(fig, "", None, plot_fname)


def compute_data_regr(aggr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_nranks = aggr_df["nranks"].unique()
    all_policies = aggr_df["policy"].unique()

    avg_data: list[dict] = []
    max_data: list[dict] = []

    for nranks in all_nranks:
        for policy in all_policies:
            df = aggr_df[(aggr_df["nranks"] == nranks) & (aggr_df["policy"] == policy)]
            dx = df["nrounds"].astype(int)
            dy_avg = df["time_avg_ms"]
            dy_max = df["time_max_ms"]

            m_avg, c_avg = linear_regression_closed_form(dx, dy_avg)
            m_max, c_max = linear_regression_closed_form(dx, dy_max)

            avg_data.append(
                {"nranks": nranks, "policy": policy, "m": m_avg, "c": c_avg}
            )
            max_data.append(
                {"nranks": nranks, "policy": policy, "m": m_max, "c": c_max}
            )

            print(f"{nranks}/{policy} (avg): m={m_avg:.2f}, c={c_avg:.2f}")
            print(f"{nranks}/{policy} (max): m={m_max:.2f}, c={c_max:.2f}")

    avg_df = pd.DataFrame(avg_data)
    max_df = pd.DataFrame(max_data)

    return (avg_df, max_df)


def plot_regr_df(df: pd.DataFrame, key: str, plot_fname: str) -> None:
    # define a col "name" that is a str: "{nranks}/{policy}"
    df["name"] = df.apply(lambda row: f"{row['nranks']}/{row['policy']}", axis=1)
    df

    dxn = df["name"]
    dyc = df["c"]
    dym = df["m"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    bar_width = 0.5
    dx = np.arange(len(dxn))

    ax.bar(dx, dyc, bar_width, label="constant comm. cost (ms)", zorder=2)
    ax.bar(dx, dym, bar_width, label="marginal comm. cost (ms)", bottom=dyc, zorder=2)

    ax.set_title(f"Topobench: BC Time (constant vs marginal, {key})")
    ax.set_xlabel("Run Type")
    ax.set_ylabel("Time (ms)")

    ax.set_xticks(dx)
    ax.set_xticklabels(dxn)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} ms"))

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")

    ax.legend()
    fig.tight_layout()

    plot_fname = f"{plot_fname}_{key}"
    PlotSaver.save(fig, "", None, plot_fname)


def run():
    plot_init()

    df_path = "figures/20240528/topobench_fake_repl.csv"
    df_path = "figures/20240528/topobench.csv"
    data = prep_data(df_path)

    plot_fname = os.path.basename(df_path).replace(".csv", "")
    plot_data(data, plot_fname)

    avg_df, max_df = compute_data_regr(data)
    plot_regr_df(avg_df, "avg", plot_fname)
    plot_regr_df(max_df, "max", plot_fname)


if __name__ == "__main__":
    run()
