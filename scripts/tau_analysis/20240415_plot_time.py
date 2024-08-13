import glob
import numpy as np
import re
from common import PlotSaver, plot_init_big as plot_init

import matplotlib.pyplot as plt


def get_times(run_dir) -> np.ndarray:
    fpath = f"{run_dir}/run/log.txt"
    data = open(fpath, "r").read().split("\n")
    lines = [l.strip() for l in data if l.startswith("cycle=")]

    cycles = []
    times = []

    for l in lines:
        mobj = re.match(r"cycle=(\d+).*wsec_step=([^\ ]+).*$", l)
        assert mobj is not None
        c = int(mobj.group(1))
        t = float(mobj.group(2))

        cycles.append(c)
        times.append(t)

    assert len(cycles) == len(times)
    assert len(cycles) == max(cycles) + 1

    times = np.array(times)

    return times


def plot_times(times):
    data_x, data_y = times
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ylim = np.percentile(data_y, 97)

    ax.plot(data_x, data_y)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.set_title("StochSubgrid - Time per Timestep")
    ax.grid()

    # ax.set_ylim([0, ylim])
    ax.set_ylim(bottom=0, top=ylim)

    fig.tight_layout()
    plot_fname = "stochsg53_times_cum"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_all_times(
    policies: list[str], all_times: list[np.ndarray], plot_name: str
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    for p, times in zip(policies, all_times):
        data_y = times.cumsum()
        # data_y = times
        data_x = range(len(data_y))
        ax.plot(data_x, data_y, label=p)

    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")

    # ax.set_ylim(bottom=0, top=0.3)
    # ax.set_xlim(left=8000, right=9000)
    ax.grid()

    plot_fname = f"{plot_name}_times_cumsum"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def plot_all_times_marginal(
    policies: list[str], all_times: list[np.ndarray], plot_name: str
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    assert policies[0] == "lpt"
    # assert(len(policies) == 3)

    lpt_cumsum = all_times[0].cumsum()
    p1_cumsum = all_times[1].cumsum()
    p1_cumsum_delta = p1_cumsum - lpt_cumsum

    # p2_cumsum = all_times[2].cumsum()
    # p2_cumsum_delta = p2_cumsum - lpt_cumsum
    #
    data_x = range(len(lpt_cumsum))

    ax.set_title("Cumulative Excess WRT LPT")

    ax.plot(data_x, p1_cumsum_delta, label=policies[1])
    # ax.plot(data_x, p2_cumsum_delta, label=policies[2])

    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")

    # ax.set_ylim(bottom=0, top=0.3)
    # ax.set_xlim(left=8000, right=9000)
    ax.grid()

    fig.tight_layout()

    plot_fname = f"{plot_name}_times_delta"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def run():
    policies = ["lpt", "hybrid", "hybrid02"]
    policies = ["lpt", "hybrid90"]
    run_dir_pref = "/mnt/ltio/parthenon-topo/stochsg.61."

    run_dirs = [f"{run_dir_pref}{p}" for p in policies]
    print(run_dirs)

    plot_name = run_dir_pref.split("/")[-1]
    plot_name = plot_name.replace(".", "")

    all_times = [get_times(run_dir) for run_dir in run_dirs]
    plot_all_times(policies, all_times, plot_name)
    plot_all_times_marginal(policies, all_times, plot_name)

    # times = get_times(run_dir)
    # times
    # plot_times(times)
    pass


if __name__ == "__main__":
    plot_init()
    run()
