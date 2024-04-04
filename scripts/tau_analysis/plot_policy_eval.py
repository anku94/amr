import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter, MultipleLocator, LogLocator
import multiprocessing
import numpy as np
import pandas as pd
import glob
import os
import re
import struct

from common import plot_init_big as plot_init, PlotSaver, label_map, get_label
from scipy.ndimage import gaussian_filter1d
from trace_reader import TraceReader, TraceOps
from typing import Dict, List


def plot_bar():
    excess_cost = [859.96, 689.58, 663.81, 540.00, 335.86]
    avg_cost = [2089.75, 2088.49, 2088.49, 2088.49, 2088.49]
    max_cost = [2949.72, 2778.07, 2752.30, 2628.49, 2424.35]
    loc_cost = [33.41, 33.28, 100.26, 295.26, 294.97]

    names = ["Contiguous/UC", "Contiguous/AC",
             "RoundRobin/AC", "SPT/AC", "LPT/AC"]

    del excess_cost[3]
    del avg_cost[3]
    del max_cost[3]
    del loc_cost[3]
    del names[3]

    data_x = np.arange(len(names))

    fig, ax = plt.subplots(1, 1)
    width = 0.25

    ax.bar(data_x + width * 0, excess_cost,
           width, label="Excess Cost", zorder=2)
    ax.bar(data_x + width * 1, avg_cost, width, label="Avg Cost", zorder=2)
    ax.bar(data_x + width * 2, max_cost, width, label="Max Cost", zorder=2)

    ax.set_xticks(data_x + width, names, rotation=10)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Time (seconds)")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x))
    ax.legend(ncol=3)

    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_minor_locator(MultipleLocator(125))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    fig.tight_layout()

    plot_fname = "policy.sim.barplot"
    PlotSaver.save(fig, "", None, plot_fname)

    pass


def plot_bar_2():
    excess_cost = [859.96, 689.58, 663.81, 540.00, 335.86]
    avg_cost = [2089.75, 2088.49, 2088.49, 2088.49, 2088.49]
    max_cost = [2949.72, 2778.07, 2752.30, 2628.49, 2424.35]
    loc_cost = [33.41, 33.28, 100.26, 295.26, 294.97]

    names = ["Contiguous/UC", "Contiguous/AC",
             "RoundRobin/AC", "SPT/AC", "LPT/AC"]

    del excess_cost[3]
    del avg_cost[3]
    del max_cost[3]
    del loc_cost[3]
    del names[3]

    data_x = np.arange(len(names))

    fig, ax = plt.subplots(1, 1)
    width = 0.5

    #  ax.bar(data_x + width*0, excess_cost, width, label="Excess Cost", zorder=2)
    #  ax.bar(data_x + width*1, avg_cost, width, label="Avg Cost", zorder=2)
    #  ax.bar(data_x + width*2, max_cost, width, label="Max Cost", zorder=2)

    ax.bar(data_x + width * 0, loc_cost, width,
           label="Locality Cost", zorder=2)

    ax.set_xticks(data_x + width * 0, names, rotation=10)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Locality Cost (Lower = Better)")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} %".format(x))
    #  ax.legend()

    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(12.5))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    fig.tight_layout()

    plot_fname = "policy.sim.barplot2"
    PlotSaver.save(fig, "", None, plot_fname)

    pass


def concat_lbsim_ilp(trace_dir: str, num_shards: int):
    files = glob.glob(trace_dir + "/lb_sim/*.csv")

    def shard_match(x): return re.match(
        r"^ilp_actual_cost_shard_(\d+)_" +
        str(num_shards) + "\.csv", x.split("/")[-1]
    )

    matching_files = [f for f in files if shard_match(f)]
    matching_files = sorted(
        matching_files, key=lambda x: int(shard_match(x).group(1)))

    all_dfs = [pd.read_csv(f, index_col=None) for f in matching_files]

    prev_df_ts_end = -1
    for df in all_dfs:
        df["ts"] += prev_df_ts_end + 1
        prev_df_ts_end = max(df["ts"])
        print(df)
    df_concat = pd.concat(all_dfs)
    print(df_concat)

    df_concat.to_csv(f"{trace_dir}/lb_sim/ilp_actual_cost.csv", index=None)


def get_loc_score_ref(line_str):
    line_arr = line_str.strip("\n,").split(",")
    line_arr = np.array([int(i) for i in line_arr])

    loc_score = 0

    for bidx in range(0, len(line_arr) - 1):
        p = line_arr[bidx]
        q = line_arr[bidx + 1]

        pn = p / 16
        qn = q / 16

        if p == q:
            pass
        elif abs(p - q) == 1:
            loc_score += 1
        elif pn == qn:
            loc_score += 2
        else:
            loc_score += 3

    disorder = loc_score
    arr_len = len(line_arr)
    norm_disorder = disorder / arr_len
    return norm_disorder


def get_loc_score(line_str):
    line_arr = line_str.strip("\n,").split(",")
    line_arr = np.array([int(i) for i in line_arr])
    disorder = np.sum(np.abs(np.diff(line_arr)))
    arr_len = len(line_arr)
    norm_disorder = disorder / arr_len
    return norm_disorder


def get_loc_scores(fpath: str):
    f = open(fpath, "r").readlines()
    assignments = f[1::2]
    loc_scores = np.array(list(map(get_loc_score_ref, assignments)))
    return loc_scores


def read_lbsim_loc(trace_dir: str):
    files = glob.glob(trace_dir + "/block_sim/*.det")
    files = [f for f in files if "shard" not in f]

    fdata = None
    with multiprocessing.Pool(8) as p:
        fdata = list(p.map(get_loc_scores, files))

    data = {}
    for f, loc_scores in zip(files, fdata):
        print(f"Reading Loc Scores: {f}")
        policy_name = get_policy_abbrev(f)
        data[policy_name] = loc_scores

    return data


def get_policy_abbrev(fpath: str):
    fname = os.path.basename(fpath)
    fname = fname.replace(".csv", "").split("_")

    abbrev_1 = fname[0]
    abbrev_2 = "".join([x[0] for x in fname[1:]])

    return f"{abbrev_1}_{abbrev_2}"


def read_lbsim(trace_dir: str):
    files = glob.glob(trace_dir + "/block_sim/*.summ")
    files = [f for f in files if "shard" not in f]

    data = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            df["max_us"] /= 1000
            df["avg_us"] /= 1000

            policy_name = get_policy_abbrev(f)
            print(f"Read policy: {policy_name}")

            data[policy_name] = df
        except Exception as e:
            print(f"FAILED: lbsim {f}")

    return data


def aggr_lbsim(lbsim_data, loc_data, keys):
    avg_cost = {}
    max_cost = {}
    loc_cost = {}

    for k in lbsim_data:
        df = lbsim_data[k]
        avg_cost[k] = df["avg_us"].sum()
        max_cost[k] = df["max_us"].sum()
        if loc_data:
            loc_cost[k] = np.mean(loc_data[k])

    avg_array = order_dict_vals(avg_cost, keys)
    max_array = order_dict_vals(max_cost, keys)
    loc_array = order_dict_vals(loc_cost, keys)

    return avg_array, max_array, loc_array


def order_dict_vals(d, keys):
    return [d[k] for k in keys]


def get_policy_names():
    names = [
        "actual_ac",
        "contiguous_ec",
        "lpt_ec",
        "kcontigimproved_ec",
        "cppiter_ec",
        "ilp_1ac",
    ]

    names = [
        "actual_ac",
        "contiguous_ec",
        "lpt_ec",
        "kcontigimproved_ec",
        "cppiter_ec",
    ]

    return names


def plot_lbsim_excess(lbsim_data):
    #  fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim([0, 200])

    ax.set_title("(Max-Avg) compute time for different policies")

    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x))

    plot_fname = "policy.sim.ts.sq"

    idx = 0
    for pol_name in get_policy_names():
        df = lbsim_data[pol_name]
        data_x = df.index
        data_y = df["max_us"] - df["avg_us"]
        ax.plot(data_x, data_y, label=pol_name, alpha=0.8)

        frame_fname = plot_fname + f"_frame{idx}"
        ax.legend(ncol=1)
        if idx == 0:
            fig.tight_layout()
        PlotSaver.save(fig, "", None, frame_fname)

        idx += 1

    PlotSaver.save(fig, "", None, plot_fname)


def plot_lbsim_time_saved_cumul(lbsim_data):
    x = lbsim_data["actual_ac"]["max_us"] - lbsim_data["lpt_ec"]["max_us"]
    y = lbsim_data["actual_ac"]["max_us"] - \
        lbsim_data["kcontigimproved_ec"]["max_us"]

    lpt_cum = (x.cumsum() / 1e3).astype(int)
    cpp_cum = (y.cumsum() / 1e3).astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(range(len(lpt_cum)), lpt_cum, label="LPT", zorder=2)
    ax.plot(range(len(cpp_cum)), cpp_cum, label="Contig++", zorder=2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Savings (s)")
    ax.set_title("Cumulative Impact of Policy on Compute (simulated)")

    ax.xaxis.set_major_formatter(lambda x, pos: "{:.0f}K".format(x / 1e3))
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}K s".format(x / 1e3))

    ylim, ymaj, ymin = 10000, 1000, 200
    xmaj = 10000
    ax.xaxis.set_major_locator(MultipleLocator(xmaj))
    ax.set_ylim([0, ylim])
    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(MultipleLocator(ymin))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    ax.legend()

    fig.tight_layout()

    plot_fname = "policy.sim.ts.cumsave"
    PlotSaver.save(fig, "", None, plot_fname)

    pass


def plot_lbsim_actual(lbsim_data: dict[str, pd.DataFrame], plot_fname: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #  fig, ax = plt.subplots(1, 1)

    cidx = 0
    for pol_name, df in lbsim_data.items():
        data_x = df.index
        data_y = df["avg_us"]
        ax.plot(data_x, data_y, linestyle="--", color=f"C{cidx}", alpha=0.3)
        cidx += 1
        break

    ylim, ymaj, ymin = 1000, 200, 40

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (ms)")
    #  ax.set_ylim([0, 80])
    ax.set_ylim([0, ylim])

    ax.set_title("Max & Avg compute time for different policies")

    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(MultipleLocator(ymin))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x))

    # plot_fname = "policy.sim.ts.wavg.sq"

    idx = 0
    for pol_name in get_policy_names():
        df = lbsim_data[pol_name]
        data_x = df.index
        data_y = df["max_us"]
        ax.plot(data_x, data_y, label=pol_name, alpha=0.8)

        frame_fname = plot_fname + f"_frame{idx}"
        ax.legend(ncol=1)
        if idx == 0:
            fig.tight_layout()
        #  PlotSaver.save(fig, "", None, frame_fname)

        idx += 1

    global trace_dir
    PlotSaver.save(fig, trace_dir, None, plot_fname)


def plot_lbsim_barplot(lbsim_data):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    names = get_policy_names()
    avg_cost = []
    max_cost = []
    excess_cost = []
    loc_cost = []

    for name in names:
        df = lbsim_data[name]
        df_avg = df["avg_us"].sum() / 1e3
        avg_cost.append(df_avg)

        df_max = df["max_us"].sum() / 1e3
        max_cost.append(df_max)

        df_excess = df_max - df_avg
        excess_cost.append(df_excess)

    print(f"Avg cost: {avg_cost}")
    print(f"Max cost: {max_cost}")
    print(f"Excess cost: {excess_cost}")

    width = 0.25
    data_x = np.arange(len(names))
    ax.bar(data_x + width * 0, avg_cost, width, label="Avg Cost", zorder=2)
    ax.bar(data_x + width * 1, max_cost, width, label="Max Cost", zorder=2)
    ax.bar(data_x + width * 2, excess_cost,
           width, label="Excess Cost", zorder=2)

    ax.set_xticks(data_x + width, names, rotation=10)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Time (seconds)")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x))
    ax.legend(ncol=3)

    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_minor_locator(MultipleLocator(125))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    fig.tight_layout()

    plot_fname = "policy.sim.barplot"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_lbsim_barplot_2(traces: List[int]):
    names = get_policy_names()

    global trace_dir_fmt
    trace_dirs = [trace_dir_fmt.format(t) for t in traces]
    all_data = [read_lbsim(t) for t in trace_dirs]
    aggr_data = [aggr_lbsim(d, names) for d in all_data]

    all_avg = np.array([x[0] for x in aggr_data])
    all_max = np.array([x[1] for x in aggr_data])

    all_avg /= 1000
    all_max /= 1000

    all_excess = all_max - all_avg

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    nbins = len(all_avg[0])
    data_x = np.array([0, 1])
    labels_x = ["Parth-Old", "Parth-New"]
    labels_policy = ["Baseline", "LPT", "Contig++"]

    total_width = 0.7
    bin_width = total_width / nbins

    for bidx in range(nbins):
        ax.bar(
            data_x + bidx * bin_width,
            all_avg.T[bidx],
            bin_width * 0.95,
            zorder=2,
            color=f"C0",
        )

        p = ax.bar(
            data_x + bidx * bin_width,
            all_excess.T[bidx],
            bin_width * 0.95,
            label=labels_policy[bidx],
            bottom=all_avg.T[bidx],
            zorder=2,
            color=f"C{bidx+1}",
        )
        bar_data = all_max.T[bidx]
        bar_labels = ["{:.0f} s".format(x) for x in bar_data]

        ax.bar_label(p, bar_labels, fontsize=12, padding=2)

    ax.set_xlabel("Policy")
    ax.set_ylabel("Simulated Compute Phase Time (s)")
    ax.set_title("Simulated Time Comparison: Parth-Old vs Parth-New")

    ax.set_xticks(data_x, labels_x)
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(200))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 8000])

    ax.legend()

    fig.tight_layout()

    fname = "policy_sim_comp_old_vs_new"
    PlotSaver.save(fig, "", None, fname)


def plot_cluster_sim_mean(trace_dir: str):
    df = pd.read_csv(
        f"{trace_dir}/cluster_sim/cluster_sim_mean.csv", header=1, index_col=None
    )
    df.columns = ["ts", "n", "k", "mean_rel_error", "max_rel_error"]

    df = df.dropna()

    fig, ax = plt.subplots(1, 1)
    ax.plot(df["ts"], df["k"], label="Number of clusters (left)",
            color="C0", alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(
        df["ts"],
        df["k"] / df["n"],
        label="Pct of blocks (right)",
        color="C1",
        alpha=0.7,
    )

    fig.legend()
    ax2.yaxis.set_major_formatter(lambda x, pos: "{:.1f}%".format(x * 100))

    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    ax.set_title("k for mean_error < 1%")
    ax.set_xlabel("timestep")
    ax.set_ylabel("Number of clusters")
    ax2.set_ylabel("Pct of total")

    ax.set_ylim([0, 100])
    ax2.set_ylim([0, 0.2])

    fig.tight_layout()
    plot_fname = "cluster.sim.mean"
    PlotSaver.save(fig, "", None, plot_fname)


def get_policy_mat(fname: str) -> np.ndarray:
    fdata = open(fname, "rb").read()

    nranks = struct.unpack("@i", fdata[0:4])
    nitems = int((len(fdata) - 4) / 8)

    print(f"Mat len: {nitems}, ranks: {nranks}")

    unpack_fmt = "@{}d".format(nitems)
    vals = struct.unpack(unpack_fmt, fdata[4:])

    mat = np.reshape(vals, (-1, nranks[0]))
    print(f"Mat shape: {mat.shape}")

    return mat


def norm_mat(mat):
    mins = np.min(mat, axis=1)
    mins = np.vstack(mins)
    mat2 = ((mat - mins) / 1000).astype(int)
    print("99 percentile: ", np.percentile(mat2, 99))
    print("95 percentile: ", np.percentile(mat2, 95))
    return mat2


def plot_policy_mat(mat, policy_name):
    # make log safe, shouldn't be a big deal
    mat = mat + 1
    bounds = np.linspace(1, 200, 100)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="max")
    print(f"Mat: {mat.min()}, {mat.max()}")
    norm = colors.LogNorm(vmin=1, vmax=200)

    pretty_name = policy_name.split("_")
    abbrev = "".join([x[0] for x in pretty_name[1:]]).upper()
    pretty_name = f"{pretty_name[0].capitalize()} ({abbrev})"

    fig = plt.figure()
    ax = fig.subplots(1, 1)

    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")
    ax.set_title(f"Excess Time (ms) for Policy: {pretty_name}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Timestep (top to bottom)")

    fig.tight_layout()

    fig.subplots_adjust(left=0.15, right=0.78)
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])

    def cax_fmt(x, pos): return "{:.0f}ms".format(x)
    fig.colorbar(im, cax=cax, format=FuncFormatter(cax_fmt))

    #  tname = trace_dir.split('/')[-1]
    global trace_dir
    fname = f"policymat.{policy_name}"
    PlotSaver.save(fig, trace_dir, None, fname)


def plot_mats_ts_cumul(mats, names):
    maxes = list(
        map(lambda m: (np.max(m, axis=1) / 1e6).cumsum().astype(int), mats))
    maxes = list(map(lambda m: (np.max(m, axis=1) / 1e3).astype(int), mats))

    fig = plt.Figure()
    ax = fig.subplots(1, 1)

    data_x = np.arange(len(maxes[0]))
    ax.plot(data_x, maxes[0] - maxes[1], label="LPT vs Baseline")
    ax.plot(data_x, maxes[0] - maxes[2], label="Contig++ vs Baseline")
    ax.plot(data_x, maxes[0] - maxes[3], label="CppIter vs Baseline")

    #  for policy, data_y in zip(names, maxes):
    #  data_x = np.arange(len(data_y))
    #  label = policy.split('/')[-1].split('_')[0]
    #  ax.plot(data_x, data_y, label=label)

    ax.set_title("Ts-wise Time Saved vs Policy")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (ms)")

    ax.set_ylim([0, 100])

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x))

    ax.legend()
    fig.tight_layout()

    fname = "policy.cumultime"
    PlotSaver.save(fig, trace_dir, None, fname)
    pass


def run_plot_policy_mats():
    global trace_dir
    policy_glob = f"{trace_dir}/block_sim/*.csv"
    policy_files = glob.glob(policy_glob)
    policy_files = [
        "contiguous_unit_cost",
        "lpt_extrapolated_cost",
        "kcontigimproved_extrapolated_cost",
        "cppiter_extrapolated_cost",
    ]

    policy_fnames = [
        f"{trace_dir}/block_sim/{x}.ranksum.csv" for x in policy_files]
    policy_mats = list(map(get_policy_mat, policy_fnames))
    norm_mats = list(map(norm_mat, policy_mats))

    for mat, policy_name in zip(norm_mats, policy_files):
        plot_policy_mat(mat, policy_name)

    plot_mats_ts_cumul(policy_mats, policy_fnames)


def humrdstr(x):
    if x < 1e3:
        return str(x)
        pass
    if x < 1e6:
        return "{:.3f}".format(x / 1e3).strip("0") + "K"
        pass
    if x < 1e9:
        return "{:.3f}".format(x / 1e6).strip("0") + "M"
        pass
    if x < 1e12:
        return "{:.3f}".format(x / 1e9).strip("0") + "B"
        pass


def plot_scalesim_log():
    scale_df_path = "/users/ankushj/repos/amr/scripts/tau_analysis/figures/20230718/scalesim.log.csv"
    df = pd.read_csv(scale_df_path)

    policy_name_map = {
        'contiguous_unit_cost': 'Baseline',
        'lpt_actual_cost': 'LPT',
        'cpp_actual_cost': 'Contiguous-DP',
        'cpp_iter_actual_cost': 'Contiguous-DP++'
    }

    df["policy"] = df_policy.apply(policy_name_map)

    df.columns

    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots(1, 1)

    policies = df["policy"].unique()

    for policy in policies:
        df_policy = df[df["policy"] == policy]
        data_x = df_policy["nblocks"]
        data_y = df_policy["iter_time"] / 1e3

        ax.plot(data_x, data_y, "-o", label=policy, zorder=2)

    ax.set_xlabel("Number of Simulation Blocks")
    ax.set_ylabel("Solution Time (ms)")
    ax.set_title("LB Policy vs Solution Time")
    ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")

    #  ax.set_ylim(bottom=1)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.4f}".format(x).strip("0") + " ms")
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humrdstr(x)))

    #  ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0, 10.0)))
    #  ax.xaxis.set_major_locator(LogLocator(base=2, numticks=20))
    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    fig.tight_layout()

    plot_fname = "scalesim.plot"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def plot_lbsim_tradeoffs(trace_dir: str):
    fig = plt.figure()
    ax = fig.subplots(1, 1)

    keys = get_policy_names()

    lbsim_data = read_lbsim(trace_dir)
    loc_data = read_lbsim_loc(trace_dir)

    avg_arr, max_arr, loc_arr = aggr_lbsim(lbsim_data, loc_data, keys)
    avg_arr = np.array(avg_arr)
    max_arr = np.array(max_arr)
    loc_arr = np.array(loc_arr) * 100
    lb_ratios = max_arr * 100 / avg_arr

    for x, y, lab in zip(loc_arr, lb_ratios, keys):
        print(f"{x} {y} {lab}")
        ax.plot(x, y, "o", label=lab, ms=13, zorder=2)

    ax.legend()
    ax.set_ylim([100, 200])
    ax.set_xlim([0, 300])
    #  ax.set_xlim(bottom=0)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f}".format(x)))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:.0f} %".format(x))
    )
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))

    ax.set_title("Load Balance vs Locality Cost")
    ax.set_xlabel("Loc Cost (lower the better)")
    ax.set_ylabel("Load Balance (% of optimal, lower = better)")

    plt.grid(visible=True, which="major", color="#999", zorder=0)
    plt.grid(visible=True, which="minor", color="#ddd", zorder=0)

    fig.tight_layout()

    plot_fname = "lbsim.tradeoffs"
    PlotSaver.save(fig, trace_dir, None, plot_fname)


def smoothen_series(series: pd.Series, sigma: int = 1):
    return gaussian_filter1d(series, sigma=sigma)


def smoothen_series_roll(series: pd.Series, window: int = 10):
    return series.rolling(window=window).mean()


def smoothen_data(data: Dict[str, pd.DataFrame], sigma: int = 1) -> Dict[str, pd.DataFrame]:
    data_smooth = {}
    for k, v in data.items():
        vcopy = v.copy()
        vcopy["avg_us"] = smoothen_series(vcopy["avg_us"], sigma)
        vcopy["max_us"] = smoothen_series(vcopy["max_us"], sigma)
        data_smooth[k] = vcopy

    return data_smooth


def smoothen_data_roll(data: Dict[str, pd.DataFrame], window: int = 10) -> Dict[str, pd.DataFrame]:
    data_smooth = {}
    for k, v in data.items():
        vcopy = v.copy()
        vcopy["avg_us"] = smoothen_series_roll(vcopy["avg_us"], window)
        vcopy["max_us"] = smoothen_series_roll(vcopy["max_us"], window)
        data_smooth[k] = vcopy

    return data_smooth


def run_plot():
    plot_init()
    #  plot_bar()
    #  plot_bar_2()
    global trace_dir
    #  trace_dir = "/mnt/ltio/parthenon-topo/profile40"
    #  run_plot_policy_mats()

    #  plot_lbsim_tradeoffs(trace_dir)
    #  return

    trace_dir = "/mnt/ltio/parthenon-topo/athenapk4"
    trace_dir = "/mnt/ltio/parthenon-topo/stochsg44"
    data = read_lbsim(trace_dir)
    data_smooth = smoothen_data(data, 4)
    data_smooth = smoothen_data_roll(data, 50)
    #  plot_lbsim(data)
    #  plot_lbsim_excess(data)
    plot_fname = "policy.sim.ts.smooth.roll"
    plot_lbsim_actual(data_smooth, plot_fname)
    plot_lbsim_time_saved_cumul(data)
    #  plot_cluster_sim_mean(trace_dir)

    return

    global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/profile{}"
    traces = [32, 37]
    plot_lbsim_barplot_2(traces)
    plot_scalesim_log()


if __name__ == "__main__":
    run_plot()
