import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy as np

from common import plot_init_big as plot_init, PlotSaver

""" mat0, mat1 = get_evt_mat(0) and get_evt_mat(1) """


def get_evt_mat(evt_code):
    mat_path = f"{trace_dir}/evt{evt_code}.mat.pickle"
    mat = pickle.loads(open(mat_path, "rb").read())
    return mat


def plot_nblocks_from_evt_mat(mat):
    mat = mat0
    blk_cnt = np.sum(~np.isnan(mat), axis=1)
    blk_cnt

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(range(len(blk_cnt)), blk_cnt)
    ax.grid()

    ax.set_title("Stochastic Subgrid: Nblocks vs Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Nblocks")

    fig.tight_layout()
    plot_fname = "nblocks"
    PlotSaver.save(fig, "", None, plot_fname)

    pass


def plot_compute_vs_ts_fem(mat):
    mat = mat0
    mat
    mat.shape
    np.percentile(mat, 99)
    mat.fillna
    mat = np.nan_to_num(mat0)
    mat = mat.T[::-1, :]

    vmin = np.percentile(mat, 28)
    vmax = np.percentile(mat, 99)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    bounds = np.linspace(vmin, vmax, 64)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="max")

    im = ax.imshow(mat, norm=norm, aspect="auto", cmap="plasma")
    ax.set_title("Block Mat: Evt 0")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("BlockID")

    ymax = mat.shape[0]
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(ymax - y)}"))
    fig.tight_layout()

    fig.subplots_adjust(left=0.15, right=0.78)
    cax = fig.add_axes([0.81, 0.12, 0.08, 0.8])
    cax_fmt = lambda x, pos: "{:.0f} ms".format(x / 1e3)
    #  cax.yaxis.set_major_formatter(FuncFormatter(cax_fmt))
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(cax_fmt))
    plot_fname = "imshow.evt0"
    PlotSaver.save(fig, "", None, plot_fname)

def plot_evt_percentiles(mat):
    pass



def run_do_something():
    mat0 = get_evt_mat(0)
    mat0
    pass


def run():
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    trace_name = "stochsg2"
    trace_dir = trace_dir_fmt.format(trace_name)
    print(trace_dir)
    pass


if __name__ == "__main__":
    plot_init()
    run()
