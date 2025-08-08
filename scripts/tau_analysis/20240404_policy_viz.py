import os

from common import plot_init_big as plot_init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import TypedDict


class Placement:
    def __init__(self, costlist: list[float], ranklist: list[int], nranks: int):
        self.costlist = costlist
        self.ranklist = ranklist
        self.nblocks = len(costlist)
        self.nranks = nranks

    def __repr__(self) -> str:
        # preview top 5 elements max
        repr_costlist = f"[{', '.join(map(str, self.costlist[:5]))}, ...]"
        repr_ranklist = f"[{', '.join(map(str, self.ranklist[:5]))}, ...]"

        repr_str = f"""Placement:
\tcostlist={repr_costlist}
\tranklist={repr_ranklist}
\tnblocks={self.nblocks}
\tnranks={self.nranks}
        """

        return repr_str


def get_mock_placement() -> Placement:
    placement = {
        "costlist": [19, 10, 6, 4, 3, 7, 5, 5],
        "ranklist": [0, 1, 1, 1, 2, 2, 2, 2],
        "nranks": 3,
    }

    return Placement(**placement)


def read_placement(fpath: str) -> Placement:
    # file format is:
    # - one space-separated row for costlist
    # - another for ranklist
    # - and then nranks
    with open(fpath, "r") as f:
        lines = f.readlines()
        costlist = list(map(float, lines[0].strip().split()))
        ranklist = list(map(int, lines[1].strip().split()))

        nranks = int(lines[2].strip())

        placement = {
            "costlist": costlist,
            "ranklist": ranklist,
            "nranks": nranks,
        }

        return Placement(**placement)


def plot_placement(placement: Placement, fname_out: str):
    fig, ax = plt.subplots(figsize=(8, 8))

    costs = np.array(placement.costlist)
    ranks = np.array(placement.ranklist)
    nranks = placement.nranks
    nblocks = costs.shape[0]
    print(f"nblocks: {nblocks}, nranks: {nranks}")

    costs_mapped = np.argsort(np.argsort(costs))
    costs_mapped = costs_mapped / len(costs_mapped)
    # colors = costs / costs.max()
    colors = costs_mapped

    bar_width = 1

    for rank in range(nranks):
        indices = np.where(ranks == rank)[0]
        rank_costs = costs[indices]
        rank_colors = colors[indices]
        costs_and_colors = list(
            sorted(zip(rank_costs, rank_colors), key=lambda x: x[0], reverse=True)
        )
        # rank_costs = np.sort(rank_costs)
        bottom = 0
        for cost, color in costs_and_colors:
            ax.bar(
                rank,
                cost,
                bottom=bottom,
                color=plt.cm.viridis(color),
                width=bar_width,
            )
            bottom += cost

    # ax.set_xticks(np.arange(nranks))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Cost")
    ax.set_xlabel("Rank")

    # ax.set_xlim([30, 40])

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*nblocks:.0f}")
    )

    # fig.show()
    fig.savefig(f"{fname_out}.png", dpi=300)


def run_suite():
    placement_dir = "/Users/schwifty/Repos/amr-data/20240404/amr-bench-out"
    plots_dir = "/Users/schwifty/Repos/amr-data/20240404/amr-bench-plots"
    os.makedirs(plots_dir, exist_ok=True)

    pex_files = os.listdir(placement_dir)
    pex_files = [f for f in pex_files if f.endswith(".pex")]

    for f in pex_files:
        print(f"Plotting {f} ...")
        placement = read_placement(f"{placement_dir}/{f}")
        file_out = f"{plots_dir}/{os.path.basename(f)}"
        plot_placement(placement, file_out)


def run():
    placement_dir = "/Users/schwifty/Repos/amr-data/20240404/amr-bench-out"
    placement_fpath = f"{placement_dir}/hybrid.powerlaw.512.1000.pex"
    placement_fpath = f"{placement_dir}/lpt.powerlaw.512.1000.pex"
    placement_fpath = "/Users/schwifty/Repos/amr-data/20240409/lb_bench_suite.hyblpt.0.2/powerlaw.alpha-2.0.nmin50.nmax100/hybrid.powerlaw.2048.16000.pex"
    placement_fpath = "/Users/schwifty/Repos/amr-data/20240409/lb_bench_suite.hyblpt.0.8/powerlaw.alpha-2.0.nmin50.nmax100/hybrid.powerlaw.2048.16000.pex"
    placement_fpath = "/Users/schwifty/Repos/amr-data/20240410/lb_bench.hyblpt.0.1.Nmin50.Nmax100/powerlaw.alpha-2.0.rep1/hybrid.powerlaw.512.2000.pex"
    placement_fpath = "/Users/schwifty/Repos/amr-data/20240410/lb_bench.hyblpt.0.1.Nmin50.Nmax100/exp.lambda0.1.rep1/hybridcppfirst.exponential.512.2000.pex"
    placement_fpath = "/Users/schwifty/Repos/amr-data/20240410/lb_bench.hyblpt.0.3.Nmin50.Nmax100/exp.lambda0.1.rep1/hybridcppfirst.exponential.512.2000.pex"
    placement = read_placement(placement_fpath)
    print(placement)
    # placement = get_mock_placement()
    dir_out = "/Users/schwifty/Repos/amr-data/20240410/placement_plots"
    os.makedirs(dir_out, exist_ok=True)
    file_out = f"{dir_out}/{os.path.basename(placement_fpath)}"
    plot_placement(placement, file_out)
    pass


if __name__ == "__main__":
    plot_init()
    run()
    # run_suite()
