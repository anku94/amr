import matplotlib.pyplot as plt 
from matplotlib import ticker

import re
import glob
import os
import yt
import numpy as np
import multiprocessing
import sys


def plot_frame(file_path: str, plt_dir: str, plot_vars: list[tuple]) -> None:
    if not os.path.exists(file_path):
        return

    file_name = file_path.split("/")[-1]
    file_name = file_name.replace(".phdf", "")
    print(f"Plotting {file_name}")

    dataset = yt.load(file_path)
    for v in plot_vars:
        s = yt.SlicePlot(dataset, "z", v)
        s.annotate_grids()
        #  s.annotate_cell_edges()

        plt_fname = f"{file_name}_{v[1]}"
        plt_fpath = f"{plt_dir}/{plt_fname}.png"
        s.save(plt_fpath)


def get_key_id(key: str) -> int:
    key_id = int(re.findall(r'00\d+', key)[0])
    return key_id


def get_timestep(key: str) -> int:
    key_id = get_key_id(key)
    timestep = 300 * key_id
    return timestep


def plot_frame_wrapper(args):
    plot_frame(*args)


def get_all_phdf(dir_path: str) -> list[str]:
    all_files = glob.glob(dir_path + "/*.phdf")
    return all_files


def get_hist(file_path: str, plot_path: str, var: tuple) -> None:
    dataset = yt.load(file_path)
    data = dataset.all_data()
    data_var = data[var]
    dmin, dmax = np.min(data_var), np.max(data_var)
    print(f"Var: {var}, Min: {dmin}, Max: {dmax}")

    # get 1000 bins between 0 and 1
    # dmin, dmax = 0, 0.001
    # bins = np.linspace(dmin, dmax, 1000)
    # hist = np.histogram(data_var, bins=bins)
    bins, hist = np.histogram(data_var, bins=10000)
    bins / sum(bins)
    # get non-zero bin vals
    norm_bins = bins / sum(bins)
    norm_bins = norm_bins[norm_bins > 0.01]
    print(len(norm_bins))
    print(norm_bins)
    # lum(bins)
    file_path
    bins
    bins[1:30]
    hist
    sum(bins[1:])



    # plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.plot(hist[:-1], bins, label=var[1])
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    file_name = file_path.split("/")[-1]
    file_name = file_name.replace(".phdf", "")
    plt_fname = f"{file_name}_{var[1]}_hist"
    plt_fpath = f"{plot_path}/{plt_fname}.png"

    fig.savefig(plt_fpath, dpi=300)


def plot_hist(data: dict[str, np.ndarray], ptile: float, plot_fpath: str) -> None:
    fig, ax = plt.subplots(1, 1)
    for k, v in data.items():
        # ax.hist(v, bins=1000, density=True, histtype="step", label=k)
        ax.ecdf(v, label=k)

    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='major', color='#bbb')
    ax.grid(which='minor', color='#ddd')
    ax.legend()

    ax.set_xlabel("Energy")
    ax.set_ylabel("Percentile")
    ax.set_title(f"CDF (Percentile: {ptile:.0f})")
    fig.tight_layout()

    print(f"Writing to {plot_fpath}")
    fig.savefig(plot_fpath, dpi=300)


def plot_hist_2(data: dict[str, np.ndarray], ptile: float, plot_fpath: str) -> None:
    fig, ax = plt.subplots(1, 1)
    # bins = np.linspace(0, 3, 30)
    bins = [0, 0.1, 0.5, 3.0]

    for k, v in data.items():
        hist, _ = np.histogram(v, bins=bins)
        # ax.hist(v, bins=25, density=False, histtype="step", label=k)
        hist += 1
        print(hist)
        ax.plot(bins[:-1], hist, label=k)

    # ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='major', color='#bbb')
    ax.grid(which='minor', color='#ddd')
    ax.legend()

    ax.set_yscale('log')

    ax.set_xlabel("Energy")
    ax.set_ylabel("Percentile")
    ax.set_title(f"CDF (Percentile: {ptile:.0f})")
    fig.tight_layout()

    print(f"Writing to {plot_fpath}")
    fig.savefig(plot_fpath, dpi=300)


def get_fnames_to_plot(exp_name: str) -> list[str]:
    dir_path = f"/mnt/ltio/parthenon-topo/{exp_name}/run"
    all_files = glob.glob(dir_path + "/*.phdf")

    fnames = [f.split("/")[-1] for f in all_files]
    fids = [int(f.split(".")[-2]) for f in fnames if 'final' not in f]

    fids_to_plot = list(range(0, 100, 15))
    fnames_to_plot = [(f, fid) for f, fid in zip(all_files, fids) if fid in fids_to_plot]
    fnames_to_plot = sorted(fnames_to_plot, key=lambda x: x[1])
    fnames_to_plot = [f[0] for f in fnames_to_plot]

    return fnames_to_plot

def get_all_data(fnames: list[str], var: tuple) -> dict[str, np.ndarray]:
    all_data = {}
    for f in fnames:
        dataset = yt.load(f)
        data = dataset.all_data()
        data_var = data[var]
        fname = os.path.basename(f).replace(".phdf", "")
        all_data[fname] = data_var

    return all_data

def get_topk_data(all_data: dict[str, np.ndarray], ptile: float) -> dict[str, np.ndarray]:
    ptile_data = {}
    for k, v in all_data.items():
        varray = np.array(v)
        cutoff = np.percentile(varray, ptile)
        ptile_data[k] = np.array(varray[varray > cutoff])

    return ptile_data

def write_hist_to_file(all_data: dict[str, np.ndarray], dir_out: str) -> None:
    os.makedirs(dir_out, exist_ok=True)

    bins = np.linspace(0, 3, 3000)
    np.savetxt(f"{dir_out}/bins.txt", bins)

    for k, v in all_data.items():
        hist, _ = np.histogram(v, bins=bins)
        hist = hist / sum(hist)
        # hist = hist[hist > 0.01]

        hist_fpath = f"{dir_out}/{k}_hist.txt"
        np.savetxt(hist_fpath, hist)


def plot_wide_bins(data: dict[str, np.ndarray], plot_fpath: str) -> None:
    keys = list(data.keys())
    keys = sorted(keys, key=get_key_id)
    timesteps = list(map(get_timestep, keys))

    bins = [0, 0.1, 0.5, 3.0]
    all_hists = []
    for k in keys:
        hist, _ = np.histogram(data[k], bins=bins)
        hist = hist / sum(hist)
        all_hists.append(hist)

    all_hists = np.array(all_hists)
    all_hists = all_hists.T

    timesteps = list(map(get_timestep, keys))

    fig, ax = plt.subplots(1, 1)
    ax.plot(timesteps, all_hists[0], label="low energy (0-0.1)")
    ax.plot(timesteps, all_hists[1], label="mid energy (0.1-0.5)")
    ax.plot(timesteps, all_hists[2], label="high energy (0.5-3.0)")

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(which='major', color='#bbb')
    ax.grid(which='minor', color='#ddd')

    ax.set_xlabel("Timestep")
    ax.set_ylabel("% of cells in energy state")
    ax.set_title("Energy distribution of Sedov Blast Wave over time")

    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()

    fig.savefig(plot_fpath, dpi=300)


def run_cdf(exp_name):
    fnames = get_fnames_to_plot(exp_name)
    print(f"Found {len(fnames)} files to plot")

    var = ("parthenon", "c.energy")
    all_data = get_all_data(fnames, var)

    plot_dir = f"/users/ankushj/repos/amr/scripts/tau_analysis/figures/20240326"
    plot_path = f"{plot_dir}/{exp_name}/hists"

    write_hist_to_file(all_data, plot_path)

    plot_fname = f"{plot_path}/{exp_name}_wide_bins.png"
    plot_wide_bins(all_data, plot_fname)
    return

    os.makedirs(plot_path, exist_ok=True)

    ptile = 95
    ptile_data = get_topk_data(all_data, ptile)
    plot_fname = f"{plot_path}/{exp_name}_cdf_{ptile}.png"
    plot_hist(ptile_data, ptile, plot_fname)


    ptile = 90
    ptile_data = get_topk_data(all_data, ptile)
    plot_fname = f"{plot_path}/{exp_name}_cdf_{ptile}.png"
    plot_hist(ptile_data, ptile, plot_fname)

    ptile = 50
    ptile_data = get_topk_data(all_data, ptile)
    plot_fname = f"{plot_path}/{exp_name}_cdf_{ptile}.png"
    plot_hist(ptile_data, ptile, plot_fname)

    plot_fname = f"{plot_path}/{exp_name}_hist_{ptile}.png"
    plot_hist_2(ptile_data, ptile, plot_fname)

    ptile = 0
    plot_fname = f"{plot_path}/{exp_name}_cdf_{ptile}.png"
    plot_hist(all_data, ptile, plot_fname)

    plot_fname = f"{plot_path}/{exp_name}_hist_{ptile}.png"
    plot_hist_2(all_data, ptile, plot_fname)


def run():
    exp_name = "blastwave01"
    run_cdf(exp_name)

if __name__ == "__main__":
    run()
