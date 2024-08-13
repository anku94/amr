import matplotlib.pyplot as plt
import numpy as np

from common import plot_init_big, PlotSaver


def read_policy_file(fname: str) -> list[list[float]]:
    f = open(fname, "r").read().split("\n")
    even_lines = f[0::2]
    odd_lines = f[1::2]

    pairs = list(zip(even_lines, odd_lines))

    all_rank_sums = []
    for idx, pair in enumerate(pairs):
        times = [float(x) for x in pair[0].split(",") if len(x)]
        ranks = [int(x) for x in pair[1].split(",") if len(x)]
        print(f"[{idx}] times: {len(times)}, ranks: {len(ranks)}")
        assert len(times) == len(ranks)

        nranks = max(ranks) + 1
        print(f"nranks: {nranks}")

        rank_sums = [0.0] * nranks
        for i in range(len(times)):
            r: int = ranks[i]
            rank_sums[r] += times[i]

        rank_sums = sorted(rank_sums, reverse=True)
        all_rank_sums.append(rank_sums)

    return all_rank_sums


def plot_rank_sum_single(rank_sum: list[float], suffix: str):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    rs_avg = np.mean(rank_sum)
    rs_max = np.max(rank_sum)

    ax.bar(range(len(rank_sum)), rank_sum, color="blue", width=1.0)

    # plot a horizontal dotted red line at rs_avg
    ax.axhline(y=rs_avg, color="red", linestyle="--", linewidth=2.0)
    ax.axhline(y=rs_max, color="red", linestyle="--", linewidth=2.0)

    ax.set_ylim(bottom=0)
    ax.grid()

    ax.set_title(f"RankSums {suffix}")
    ax.set_ylabel("Time (us)")
    ax.set_xlabel("Rank ID")

    fig.tight_layout()

    plt_fname = f"ranksums_{suffix}"
    PlotSaver.save(fig, "", None, plt_fname)


def run_policy(trace_dir: str, policy_name: str):
    policy_file = f"{trace_dir}/block_sim/{policy_name}.det"

    all_rank_sums = read_policy_file(policy_file)
    ntoskip = 189
    ntoplot = 2

    slice_to_plot = all_rank_sums[ntoskip:ntoskip + ntoplot]
    for i in range(0, ntoplot):
        suffix = f"{policy_name}_{i}"
        plot_rank_sum_single(slice_to_plot[i], suffix)
        break


def run():
    trace_name = "stochsg.57.hybrid90"
    trace_dir = f"/mnt/ltio/parthenon-topo/{trace_name}"

    # policy_name = "hybrid30"
    # run_policy(trace_dir, policy_name)
    #
    # policy_name = "hybrid50"
    # run_policy(trace_dir, policy_name)
    #
    # policy_name = "hybrid70"
    # run_policy(trace_dir, policy_name)
    #
    policy_name = "hybrid90"
    run_policy(trace_dir, policy_name)

    policy_name = "lpt"
    run_policy(trace_dir, policy_name)

    policy_name = "cdpi50"
    run_policy(trace_dir, policy_name)


if __name__ == "__main__":
    plot_init_big()
    run()
