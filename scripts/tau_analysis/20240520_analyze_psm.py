import glob
import struct
import os

import matplotlib.pyplot as plt
from matplotlib import ticker

import multiprocessing
import numpy as np

from common import plot_init_big, PlotSaver


def get_files(dir_path: str) -> list[str]:
    all_files = glob.glob(dir_path + "/trace/latency*")
    return all_files


def parse_file(fpath: str) -> np.ndarray:
    fbytes = open(fpath, "rb").read()

    s = struct.Struct("<bQQ")
    unpacked = s.iter_unpack(fbytes)

    # we want to calculate all latencies
    all_lat: list[int] = []
    msg_dict: dict[str, int] = {}

    ntotal = 0
    nunpaired_acks = 0

    for line in unpacked:
        is_ack, timestamp, msgid = line
        ntotal += 1

        if msgid in msg_dict and is_ack == 1:
            all_lat.append(timestamp - msg_dict[msgid])
            del msg_dict[msgid]
        elif is_ack == 1:
            nunpaired_acks += 1
        else:
            msg_dict[msgid] = timestamp

    print(f"Total messages: {ntotal}, unpaired acks: {nunpaired_acks}")
    return np.array(all_lat)


def parse_run(dir_path: str) -> list[int]:
    all_files = get_files(dir_path)
    all_lat: list[int] = []

    with multiprocessing.Pool(16) as pool:
        all_lat = pool.map(parse_file, all_files)

    all_lat = np.concatenate(all_lat)
    return all_lat


def plot_lat_hists(all_lats: list[np.ndarray], all_runs: list[str]):
    run_name = None

    all_lats
    all_runs

    run_dict = {}
    for run, lats in zip(all_runs, all_lats):
        run_name = os.path.basename(run)
        run_type = run_name.split(".")[-1]
        run_name = ".".join(run_name.split(".")[:-1])
        run_dict[run_type] = lats

    assert run_name is not None
    plot_order = ["baseline", "cdp", "hybrid30", "hybrid90", "lpt"]

    def plot_inner(**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for run in plot_order:
            if run not in run_dict:
                continue

            data_lat = run_dict[run]
            # ax.hist(data_lat, bins=1000, alpha=0.9, label=run, histtype="step", cumulative=True)
            ax.hist(data_lat, label=run, **kwargs)

        ax.set_xlabel("Latency (us)")
        ax.set_ylabel("Count")
        ax.set_title("P2P Message Latencies (512 ranks)")
        # ax.set_xlim([0, 20000])

        ax.xaxis.grid(which="major", color="#bbb")
        ax.yaxis.grid(which="major", color="#bbb")
        ax.yaxis.grid(which="minor", color="#ddd")

        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M")
        )

        ax.legend()

        # fig.tight_layout()

        return fig, ax

    def plot_inner_cdf(**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for run in plot_order:
            if run not in run_dict:
                continue

            data_lat = run_dict[run]
            # ax.hist(data_lat, bins=1000, alpha=0.9, label=run, histtype="step", cumulative=True)

            counts, bin_edges = np.histogram(data_lat, bins=1000, density=True)
            cdf = np.cumsum(counts * np.diff(bin_edges))
            ax.plot(bin_edges[1:], cdf, label=run, **kwargs)

        ax.set_xlabel("Latency (us)")
        ax.set_ylabel("Percentile")
        ax.set_title("P2P Message Latencies (512 ranks)")
        # ax.set_xlim([0, 20000])

        ax.xaxis.grid(which="major", color="#bbb")
        ax.yaxis.grid(which="major", color="#bbb")
        ax.yaxis.grid(which="minor", color="#ddd")

        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "{:.0f}%".format(x * 100))
        )

        ax.legend()

        # fig.tight_layout()

        return fig, ax

    def plot_inner_cdf_transposed(**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for run in plot_order:
            if run not in run_dict:
                continue

            data_lat = run_dict[run]
            counts, bin_edges = np.histogram(data_lat, bins=1000, density=True)
            cdf = np.cumsum(counts * np.diff(bin_edges))
            ax.plot(cdf, bin_edges[1:], label=run, **kwargs)

        ax.set_ylabel("Latency (us)")
        ax.set_xlabel("Percentile")
        ax.set_title("P2P Message Latencies (512 ranks)")

        ax.yaxis.grid(which="major", color="#bbb")
        ax.yaxis.grid(which="minor", color="#ddd")
        ax.xaxis.grid(which="major", color="#bbb")
        ax.xaxis.grid(which="minor", color="#ddd")

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f} ms"))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "{:.0f}%".format(x * 100))
        )

        ax.legend()
        fig.tight_layout()

        return fig, ax

    fig, ax = plot_inner(bins=1000, alpha=0.9, histtype="step", cumulative=False)
    fig.tight_layout()
    plot_fname = f"psm_lat_hist_{run_name}_full"
    PlotSaver.save(fig, "", None, plot_fname)

    ax.set_xlim(left=0, right=20000)
    fig.tight_layout()
    plot_fname = f"psm_lat_hist_{run_name}_zoom"
    PlotSaver.save(fig, "", None, plot_fname)

    fig, ax = plot_inner(bins=1000, alpha=0.9, histtype="step", cumulative=True)
    ax.set_xlim(left=0, right=20000)
    fig.tight_layout()
    plot_fname = f"psm_lat_hist_{run_name}_zoom_cum"
    PlotSaver.save(fig, "", None, plot_fname)

    fig, ax = plot_inner(
        bins=1000, alpha=0.9, histtype="step", cumulative=True, density=True
    )
    ax.set_xlim(left=0, right=20000)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    plot_fname = f"psm_lat_hist_{run_name}_zoom_cum_norm"
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.grid(which="minor", color="#ddd")
    PlotSaver.save(fig, "", None, plot_fname)

    plot_fname = f"psm_lat_hist_{run_name}"
    PlotSaver.save(fig, "", None, plot_fname)

    fig, ax = plot_inner_cdf()
    plot_fname = f"psm_lat_cdf_{run_name}"
    PlotSaver.save(fig, "", None, plot_fname)

    ax.set_xlim(left=0, right=20000)
    fig.tight_layout()
    plot_fname = f"psm_lat_cdf_{run_name}_zoom"
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.grid(which="minor", color="#ddd")
    PlotSaver.save(fig, "", None, plot_fname)

    fig, ax = plot_inner_cdf_transposed()
    plot_fname = f"psm_lat_cdf_{run_name}_transposed"
    PlotSaver.save(fig, "", None, plot_fname)

    ax.set_xlim(left=0, right=0.99)
    ax.set_ylim(bottom=0, top=20000)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    fig.tight_layout()
    plot_fname = f"psm_lat_cdf_{run_name}_transposed_zoom"
    PlotSaver.save(fig, "", None, plot_fname)


def run():
    dir_path = "/users/ankushj/repos/amr-workspace/amr-psmhack-prefix/scripts/psm_lat"
    all_files = get_files(dir_path)
    latencies = parse_file(all_files[0])

    latencies

    run_dir = "/mnt/ltio/parthenon-topo/blastw512.psmlat.01.lpt"
    lats = parse_run(run_dir)
    lats
    len(lats)

    run_dir_pref = "/mnt/ltio/parthenon-topo/blastw512.psmlat.01.*"
    all_run_dirs = glob.glob(run_dir_pref)
    all_run_dirs
    all_lats = [parse_run(run_dir) for run_dir in all_run_dirs]

    all_runs = all_run_dirs
    plot_lat_hists(all_lats, all_runs)


if __name__ == "__main__":
    plot_init_big()
    run()
