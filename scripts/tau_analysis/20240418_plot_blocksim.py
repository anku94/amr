import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from common import plot_init_big as plot_init, PlotSaver


def get_nblocks(trace_dir: str, policy_name: str) -> int:
    policy_file = f"{trace_dir}/block_sim/{policy_name}.det"
    data = open(policy_file, "r").read().split("\n")
    data = data[0::2]

    list_nblocks = list(map(lambda x: len(x.split(",")), data))
    list_nblocks = np.array(list_nblocks) - 1
    return list_nblocks


def get_policy_cumsum(trace_dir: str, policy_name: str) -> np.ndarray:
    policy_file = f"{trace_dir}/block_sim/{policy_name}.summ"
    df = pd.read_csv(policy_file)
    max_us = df["max_us"].cumsum()
    return max_us


def plot_policies(policy_names: list[str], policy_data: list[np.ndarray]):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    for name, data in zip(policy_names, policy_data):
        ax.plot(data, label=name)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time")
    ax.grid()

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend()
    fig.tight_layout()

    plot_fname = "policy_sim"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_policies_rel(policy_names: list[str], policy_data: list[np.ndarray]):
    idx_lpt = policy_names.index("lpt")
    data_lpt = policy_data[idx_lpt]

    idxes_nonlpt = [i for i in range(len(policy_names)) if i != idx_lpt]
    names_nonlpt = [policy_names[i] for i in idxes_nonlpt]
    data_nonlpt = [policy_data[i] for i in idxes_nonlpt]

    data_nonlpt_rel = [d - data_lpt for d in data_nonlpt]
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    for name, data in zip(names_nonlpt, data_nonlpt_rel):
        data = data / 1e6
        ax.plot(data, label=name)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time")
    ax.grid()

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend()
    fig.tight_layout()

    plot_fname = "policy_sim_rel"
    PlotSaver.save(fig, "", None, plot_fname)


def plot_nblocks_per_rank(nblocks_per_rank: np.ndarray):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    ax.plot(nblocks_per_rank)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Blocks per Rank")
    ax.grid()

    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    plot_fname = "nbpr"
    PlotSaver.save(fig, "", None, plot_fname)


def run():
    trace_name = "stochsg.58.lpt"
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    trace_dir = trace_dir_fmt.format(trace_name)

    policies = ["lpt", "hybrid90"]
    nblocks = get_nblocks(trace_dir, policies[1])
    nblocks_per_rank = nblocks / 512

    plot_nblocks_per_rank(nblocks_per_rank)
    # data = [get_policy_cumsum(trace_dir, p) for p in policies]
    #
    # plot_policies(policies, data)
    # plot_policies_rel(policies, data)


if __name__ == "__main__":
    plot_init()
    run()
