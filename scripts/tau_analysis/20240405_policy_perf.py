import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import plot_init_big as plot_init


def plot_run(fpath: str, plot_path: str) -> None:
    df = pd.read_csv(fpath)

    nblocks = len(df["nblocks"].unique())

    df["rel_off"] = df["time_max"] / df["time_avg"]
    df["rel_off_sq"] = df["rel_off"] ** 2
    print(df)
    # df["rel_off_sq"] /= nblocks

    aggr_df = df.groupby(["policy", "nranks"], as_index=False).agg(
        {"rel_off_sq": ["sum", "count"]}
    )
    aggr_df.columns = ["policy", "nranks", "rel_off_sq", "rel_off_sq_count"]
    aggr_df["rel_off_sq"] /= aggr_df["rel_off_sq_count"]
    print(aggr_df)
    aggr_df["rel_off"] = aggr_df["rel_off_sq"] ** 0.5
    # print(df)

    # pivot aggr_df on policy and nranks
    aggr_df_pivot = aggr_df.pivot(index="policy", columns="nranks", values="rel_off")

    policy_order = [
        "Baseline",
        "kContigImproved",
        "kContig++Iter_50",
        "kContig++Iter_250",
        "LPT",
        "Hybrid",
        "HybridCppFirst",
        "HybridCppFirstV2",
    ]

    policy_map = {
        "Baseline": "Baseline",
        "kContigImproved": "CDP",
        "kContig++Iter_50": "CDP+I50",
        "kContig++Iter_250": "CDP+I250",
        "LPT": "LPT",
        "HybridCppFirstV2": "Hybrid",
    }

    aggr_df_pivot["policy_cat"] = pd.Categorical(
        aggr_df_pivot.index, categories=policy_order, ordered=True
    )
    aggr_df_pivot = aggr_df_pivot.sort_values("policy_cat")
    aggr_df_pivot.drop(columns=["policy_cat"], inplace=True)

    print(aggr_df_pivot)
    # print(df)

    dir_path = os.path.dirname(fpath)
    policy_params, run_params = run_parse_params(dir_path)
    plot_title = f"Policy Performance: {policy_params} \n {run_params}"

    fig, ax = plt.subplots(figsize=(9, 8))
    for index, row in aggr_df_pivot.iterrows():
        if index not in policy_map:
            continue
        label = policy_map[index]
        index_str = str(index)
        is_hybrid = index_str.startswith("Hybrid")
        if index == "LPT":
            marker = "s"
        elif is_hybrid:
            marker = "D"
        else:
            marker = "o"
        ax.plot(row, label=label, marker=marker)
    ax.legend()

    ax.set_xlabel("Number of Ranks")
    ax.set_ylabel("Relative CLB Performance (%, lower=better)")
    ax.set_title(plot_title)

    # format y ticks as percentage
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # gridlines
    ax.yaxis.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")
    ax.xaxis.grid(which="major", color="#bbb")
    ax.set_ylim(1.0, 2.0)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)


def run_parse_params(path: str) -> tuple[str, str]:
    dir_name = os.path.basename(os.path.dirname(path))

    distrib_params: str = "Unknown"
    policy_params: str = "Unknown"

    print(dir_name)
    mobj = re.match("^.*hyblpt\.([0-9\.]+)\.Nmin(\d+).Nmax(\d+)$", dir_name)
    assert mobj is not None
    hyb_lpt_threshold = float(mobj.group(1))
    nmin = int(mobj.group(2))
    nmax = int(mobj.group(3))

    policy_params = f"hybrid_lpt_pct={hyb_lpt_threshold*100:.0f}%"

    run_name = os.path.basename(path)
    print(run_name)
    gauss_regex = "^gaussian\.mean([0-9]+)\.stddev([0-9]+)\.rep.*$"
    mobj = re.match(gauss_regex, run_name)
    if mobj:
        gauss_mean = int(mobj.group(1))
        gauss_stddev = int(mobj.group(2))
        distrib_params = f"gaussian (mean={gauss_mean}, std={gauss_stddev})"

    powerlaw_regex = "^powerlaw\.alpha-([0-9\.]+)\.rep.*$"
    mobj = re.match(powerlaw_regex, run_name)
    if mobj:
        powerlaw_alpha = float(mobj.group(1))
        distrib_params = f"powerlaw (alpha={powerlaw_alpha}"

    exp_regex = "exp.lambda([0-9\.]+)\.rep.*$"
    mobj = re.match(exp_regex, run_name)
    if mobj:
        exp_lambda = float(mobj.group(1))
        distrib_params = f"exponential (lambda={exp_lambda})"

    distrib_params = f"{distrib_params} [Nmin={nmin}, Nmax={nmax}]"

    print(policy_params)
    print(distrib_params)
    return (policy_params, distrib_params)


def run_plot(run_dir: str, plot_dir: str) -> None:
    plot_subdir = f"{plot_dir}/{os.path.basename(run_dir)}"
    os.makedirs(plot_subdir, exist_ok=True)

    # get all directories in run_dir
    run_dirs = os.listdir(run_dir)
    run_dirs = [f"{run_dir}/{d}" for d in run_dirs if os.path.isdir(f"{run_dir}/{d}")]
    print(run_dirs)

    for r in run_dirs:
        run_csv = f"{r}/benchmark.csv"
        plot_name = f"{plot_subdir}/{os.path.basename(r)}.png"
        plot_run(run_csv, plot_name)
    pass


def run_all():
    run_root = "/Users/schwifty/Repos/amr-data/20240411/lb_bench_v2"

    run_dir = f"{run_root}.hyblpt.0.1.Nmin50.Nmax100"
    run_plot(run_dir, f"{run_root}/plots")

    run_dir = f"{run_root}.hyblpt.0.2.Nmin50.Nmax100"
    run_plot(run_dir, f"{run_root}/plots")

    run_dir = f"{run_root}.hyblpt.0.3.Nmin50.Nmax100"
    run_plot(run_dir, f"{run_root}/plots")


def run():
    df_dir = "/Users/schwifty/Repos/amr-data/20240404/amr-bench-out"
    df_path = f"{df_dir}/benchmark.csv"

    # prep_data(df_path)
    run_root = "/Users/schwifty/Repos/amr-data/20240410"
    run_dir = f"{run_root}/lb_bench.hyblpt.0.1.Nmin50.Nmax100"
    run = f"{run_dir}/gaussian.mean75.stddev10.rep1"

    plot_dir = f"{run_root}/plots"
    # run_plot(run_dir, plot_dir)
    # run = f"{run_root}/lb_bench_suite.hyblpt.0.2/powerlaw.alpha-3.0.nmin50.nmax100"
    run_csv = f"{run}/benchmark.csv"
    run_parse_params(run)

    plot_name = f"{run_root}/plots/{os.path.basename(run)}.png"
    os.makedirs(os.path.dirname(plot_name), exist_ok=True)

    plot_run(run_csv, plot_name)
    pass


if __name__ == "__main__":
    plot_init()
    run_all()
    # run()
