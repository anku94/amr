import collections

import colorcet
import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn import linear_model

from trace_reader import TraceReader, TraceOps


def plot_scatter_ds(data_x, data_y, label_x, label_y, plot_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    df = pd.DataFrame({"x": data_x, "y": data_y})

    cvs = ds.Canvas(plot_width=1024, plot_height=1024)
    agg = cvs.points(df, "x", "y")
    img = ds.tf.shade(agg).to_pil()

    xmin = min(data_x)
    xmax = max(data_x)
    ymin = min(data_y)
    ymax = max(data_y)

    plt.imshow(img, aspect="auto", extent=[xmin, xmax, ymin, ymax])

    ax.set_xlabel("Variable: ({})".format(label_x))
    ax.set_ylabel("Variable: ({})".format(label_y))
    ax.set_title("{} vs {} for 30568 * 512 tasks".format(label_y, label_x))

    #  ax.xaxis.set_major_formatter(lambda x, pos: "{:.1f} s".format(x / 1e6))
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f} s".format(x / 1e6))

    plot_path = "{}/scatter_{}_vs_{}.pdf".format(
        plot_dir, label_y.lower(), label_x.lower()
    )

    fig.tight_layout()

    fig.savefig(plot_path, dpi=600)


def plot_scatter_ds_msg(data_x, data_y, label_x, label_y, plot_dir) -> None:
    xmin = min(data_x) - 6
    xmax = max(data_x)
    ymin = min(data_y)
    ymax = max(data_y)

    fig, ax = plt.subplots(1, 1)

    df = pd.DataFrame({"x": data_x, "y": data_y})

    cvs = ds.Canvas(
        plot_width=1024, plot_height=1024, x_range=(xmin, xmax), y_range=(ymin, ymax)
    )

    agg = cvs.points(df, "x", "y")
    img = ds.tf.shade(agg, cmap=colorcet.fire).to_pil()

    print(collections.Counter(data_x))

    plt.imshow(img, aspect="auto", extent=[xmin, xmax, ymin, ymax])

    ax.set_xlabel("Variable: ({})".format(label_x))
    ax.set_ylabel("Variable: ({})".format(label_y))
    ax.set_title("{} vs {} for 30568 * 512 tasks".format(label_y, label_x))

    #  ax.xaxis.set_major_formatter(lambda x, pos: "{:.1f} s".format(x / 1e6))
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f} s".format(x / 1e6))

    plot_path = "{}/scatter_{}_vs_{}.pdf".format(
        plot_dir, label_y.lower(), label_x.lower()
    )

    fig.tight_layout()

    fig.savefig(plot_path, dpi=600)


def plot_scatter(data_x, data_y, label_x, label_y, plot_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    ax.scatter(data_x, data_y)
    ax.set_xlabel("Independent Variable ({})".format(label_x))
    ax.set_ylabel("Dependent Variable ({})".format(label_y))
    ax.set_title("{} vs {} for 30568 * 512 tasks".format(label_y, label_x))

    plot_path = "{}/scatter_{}_vs_{}.pdf".format(
        plot_dir, label_y.lower(), label_x.lower()
    )

    fig.savefig(plot_path, dpi=600)


def plot_scatter_standardize(data_x, data_y, label_x, label_y, plot_dir) -> None:
    min_ts = min(len(data_x), len(data_y))
    data_x = data_x[:min_ts]
    data_y = data_y[:min_ts]

    print(data_x.shape, data_y.shape)
    data_x = data_x.ravel()
    data_y = data_y.ravel()
    print(data_x.shape, data_y.shape)

    #  plot_scatter_ds(data_x, data_y, label_x, label_y, plot_dir)
    plot_scatter_ds_msg(data_x, data_y, label_x, label_y, plot_dir)


def run_plot_evt_vs_msgcnt(trace_dir: str, plot_dir: str) -> None:
    tr = TraceReader(trace_dir)
    TraceOps.trace = tr

    all_evts = ["AR1", "AR2", "AR3", "AR3_UMBT", "SR"]

    evty_label = "tau:AR1+AR2+SR"
    evty_label = "tau:AR3-AR3_UMBT"
    evty_label = "tau:SR"
    evty_mat = TraceOps.multimat(evty_label)
    #  evty_label = "AR3_UMBT"
    #  evty_label = "AR3"
    evtx_label = "msgcnt:LoadBalancing"
    evtx_label = "npeer:BoundaryComm"

    #  msg_mat = tr.get_msg_count("FluxExchange")
    #  print(msg_mat.shape)
    evtx_mat = multimat(evtx_label)

    #  --- Fixed code for AR3 - AR3_UMBT
    #  evty1_label = "tau:AR3"
    #  evty1_mat = multimat(evty_label)
    #  evty2_label = "tau:AR3_UMBT"
    #  evty2_mat = multimat(evty2_label)
    #  evty_label = "tau:AR3subUMBT"
    #  evty_mat = evty1_mat - evty2_mat

    plot_scatter_standardize(evtx_mat, evty_mat, evtx_label, evty_label, plot_dir)


def run_plot():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    plot_dir = "figures/regression"

    run_plot_evt_vs_msgcnt(trace_dir, plot_dir)


def read_regr_mats(all_labels):
    all_mats = list(map(TraceOps.multimat, all_labels))
    all_lens = map(lambda x: x.shape[0], all_mats)
    min_len = min(all_lens)
    all_mats_clipped = [i[:min_len] for i in all_mats]
    all_mats_raveled = [m.ravel() for m in all_mats_clipped]
    return all_mats_raveled


def run_regr_actual(X, y):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    print(regr.score(X, y))
    print(np.array(regr.coef_, dtype=int))
    print(regr.intercept_)


def run_regr_raveled(evtx_label, evty_label):
    rx, ry = read_regr_mats([evtx_label, evty_label])

    #  X = np.array([rx, rx*rx])
    X = np.array([rx])
    X = X.transpose()
    y = ry

    run_regr_actual(X, y)


def run_regr_phases():
    all_labels = [
        "tau:AR1",
        "tau:AR2",
        "tau:SR",
        "tau:AR3",
        "tau:AR3_UMBT",
        "rcnt:",
        "msgcnt:FluxExchange",
        "msgcnt:BoundaryComm",
        "msgcnt:LoadBalancing",
        "tau:AR3-AR3_UMBT",
    ]

    all_mats = read_regr_mats(all_labels)

    def run(mat_idx):
        selected_mats = [all_mats[i] for i in mat_idx]
        X = np.array(selected_mats[:-1])
        y = selected_mats[-1]

        run_regr_actual(X.transpose(), y)

    # repeated relations for y are eliminating least correlated
    # AR1
    mat_idx = [5, 6, 7, 8, 0]
    mat_idx = [5, 6, 7, 0]

    # AR2
    mat_idx = [5, 6, 7, 8, 1]
    mat_idx = [5, 6, 7, 1]

    # SR
    mat_idx = [5, 6, 7, 8, 2]
    mat_idx = [5, 6, 7, 2]

    # AR3
    mat_idx = [5, 6, 7, 8, 3]

    # AR3_UMBT
    mat_idx = [5, 6, 7, 8, 4]

    # AR3 - AR3_UMBT
    mat_idx = [5, 6, 7, 8, 9]
    mat_idx = [8, 9]
    run(mat_idx)


def run_regression():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    TraceOps.trace = TraceReader(trace_dir)

    evtx_label = "rcnt:"
    evty_label = "tau:AR1"
    evty_label = "tau:AR2"
    evty_label = "tau:SR"
    evty_label = "tau:AR3"
    evty_label = "tau:AR3_UMBT"
    evty_label = "tau:AR3-AR3_UMBT"

    run_regr_raveled(evtx_label, evty_label)


if __name__ == "__main__":
    run_plot()
