import pandas as pd

from common import PlotSaver, plot_init_big
from matplotlib import pyplot as plt


def get_nblocks_mat(trace_dir: str):
    df_path = f"{trace_dir}/prof.aggr.evt3.csv"
    df = pd.read_csv(df_path)
    aggr_df = df.groupby(["sub_ts", "rank"]).size().reset_index(name="nblocks")
    mat = aggr_df.pivot(index="sub_ts", columns="rank", values="nblocks").to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Timestep")
    ax.set_title("Nblocks Per-rank Per-timestep")

    fig.tight_layout()

    fig.subplots_adjust(left=0.15, right=0.8)
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])
    cbar = fig.colorbar(im, cax=cax)

    plot_fname = "nblocks_mat_byts"
    PlotSaver.save(fig, trace_dir, None, plot_fname)


def run():
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    trace_name = "stochsg21"
    trace_dir = trace_dir_fmt.format(trace_name)
    get_nblocks_mat(trace_dir)


if __name__ == "__main__":
    plot_init_big()
    run()
