import collections

import colorcet
import datashader as ds
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn import linear_model

from common import label_map
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
    evty_mat = tr.multimat(evty_label)
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


def plot_box_load_vs_evt(data_x, evt_x, data_y, evt_y, plot_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    def evt_to_nice(evt):
        return label_map[evt]

    def evtnice_to_flab(evtnice):
        file_label = re.sub(r"\$|\{|\}|\(|\)", "", evtnice)
        file_label = re.sub(r"\ ", "_", file_label)
        file_label = file_label.lower()
        return file_label

    def evt_to_label(evt, evtnice):
        evt_to_label = {
            "tau": "Phase Time {}",
            "msgcnt": "Message Count {}",
            "msgsz": "Message Size {}",
            "npeer": "Unique Peer Count {}",
            "rcnt": "Load (MeshBlock Count) {}",
        }

        evtpref = evt.split(":")[0]
        fmtstr = evt_to_label[evtpref]
        return fmtstr.format(evtnice)

    if "tau" in evt_x:
        ax.xaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))

    if "tau" in evt_y:
        ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} ms".format(x / 1e3))

    evtnice_x = evt_to_nice(evt_x)
    evtnice_y = evt_to_nice(evt_y)
    flab_x = evtnice_to_flab(evtnice_x)
    flab_y = evtnice_to_flab(evtnice_y)
    label_x = evt_to_label(evt_x, evtnice_x)
    label_y = evt_to_label(evt_y, evtnice_y)

    plot_title = "{} vs {}".format(evtnice_x, evtnice_y)
    plot_path = "{}/{}_vs_{}.pdf".format(plot_dir, flab_x, flab_y)

    """ manually enable if data_x is something other than rcnt,
    and it needs to be rounded off for a lower number of box plots
    """
    #  data_x = np.around(data_x / 100) * 100
    #  ax.xaxis.set_major_formatter(lambda x, pos: "{:.0f}".format(x * 100))

    data_x = data_x.astype(int)
    df = pd.DataFrame({"x": data_x, "y": data_y})
    df = df.sort_values("x").groupby("x").agg(list)
    data = df["y"].to_list()
    data = [np.array(i) for i in data]

    ax.boxplot(data, showfliers=False)

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(plot_title)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)


def plot_load_vs_evt(load, evt, evt_label, plot_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    #  evt_label = evt_label.replace(":", "_")
    evt_label = label_map[evt_label]
    file_label = re.sub("\$|\{|\}", "", evt_label)

    data_x = load
    label_x = "Load (MeshBlock Count)"
    data_y = evt
    label_y = "Phase Time {}".format(evt_label)
    plot_title = "Load vs {}".format(evt_label)
    plot_path = "{}/load_vs_{}.pdf".format(plot_dir, file_label)

    xmin = min(data_x) - 1
    xmax = max(data_x)
    ymin = min(data_y)
    ymax = max(data_y)

    df = pd.DataFrame({"x": data_x, "y": data_y})

    cvs = ds.Canvas(
        plot_width=1024, plot_height=1024, x_range=(xmin, xmax), y_range=(ymin, ymax)
    )
    agg = cvs.points(df, "x", "y")
    #  img = ds.tf.shade(agg).to_pil()
    img = ds.tf.shade(ds.tf.dynspread(agg, threshold=0.1, max_px=10)).to_pil()

    plt.imshow(img, aspect="auto", extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(plot_title)

    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f} ms".format(x / 1e3))
    ax.set_xticks(range(xmin, xmax))

    fig.tight_layout()
    fig.savefig(plot_path, dpi=600)


def plot_load(load_mat, lb_idxes, plot_dir: str) -> None:
    plot_path = "{}/load_vs_ts.pdf".format(plot_dir)

    fig, ax = plt.subplots(1, 1)

    load_min = np.min(load_mat, axis=1)
    load_mean = np.mean(load_mat, axis=1)
    load_max = np.max(load_mat, axis=1)

    num_ts = load_mat.shape[0]

    data_x = range(num_ts)

    ax.plot(data_x, load_min, label="Min Load")
    ax.plot(data_x, load_mean, label="Avg Load")
    ax.plot(data_x, load_max, label="Max Load")

    ax.plot(lb_idxes, [0] * lb_idxes, "ro", markersize=4)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Load (Meshblock Count)")

    ax.legend()

    ax.set_title("Load Distribution Across Timesteps")
    fig.savefig(plot_path, dpi=600)
    plt.close(fig)


def run_plot_load_vs_evt(trace_dir: str, plot_dir: str) -> None:
    all_labels = [
        "tau:AR1",
        "tau:AR2",
        "tau:SR",
        "tau:AR3",
        "tau:AR3_UMBT",
        "rcnt:",
        "msgcnt:BoundaryComm",
        "msgsz:BoundaryComm",
        "npeer:BoundaryComm",
    ]

    tr = TraceOps(trace_dir)
    all_rmats = read_regr_mats(tr, all_labels)

    load = all_rmats[5]
    evt_x = load
    evtlab_x = "rcnt:"

    for evt_to_plot in [0, 1, 2, 3, 4, 6, 7, 8]:
        plot_box_load_vs_evt(
            load, all_rmats[evt_to_plot], all_labels[evt_to_plot], plot_dir
        )

    evtidx_x = 6
    evtidx_y = 2
    plot_box_load_vs_evt(
        all_rmats[evtidx_x],
        all_labels[evtidx_x],
        all_rmats[evtidx_y],
        all_labels[evtidx_y],
        plot_dir
    )

    return

    label = "tau:AR3-AR3_UMBT"
    mat = all_rmats[3] - all_rmats[4]
    plot_load_vs_evt(load, mat, label, plot_dir)

    mat_rcnt = all_mats[5]

    lb_idxes = get_lb_timesteps(tr)
    plot_load(mat_rcnt, lb_idxes, plot_dir)
    pass


def run_plot():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    plot_dir = "figures/20220809"

    #  run_plot_evt_vs_msgcnt(trace_dir, plot_dir)
    run_plot_load_vs_evt(trace_dir, plot_dir)


def read_regr_mats(tr, all_labels, idx_drop=None):
    all_mats_orig = list(map(tr.multimat, all_labels))
    all_mats = all_mats_orig

    all_lens = list(map(lambda x: x.shape[0], all_mats))
    min_len = min(all_lens)

    if idx_drop is not None:
        idx_drop = idx_drop[idx_drop < min_len]
        all_mats = list(map(lambda x: np.delete(x, idx_drop, axis=0), all_mats_orig))

    all_lens = map(lambda x: x.shape[0], all_mats)
    min_len = min(all_lens)
    all_mats_clipped = [i[:min_len] for i in all_mats]
    all_mats_raveled = [m.ravel() for m in all_mats_clipped]
    return all_mats_raveled


def get_lb_timesteps(tr):
    label = "tau:AR3-AR3_UMBT"
    lb_mat = tr.multimat(label)
    lb_mean = np.mean(lb_mat, axis=1)
    lb_idxes = np.argwhere(lb_mean > 1000).ravel()

    return lb_idxes


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
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    tr = TraceOps(trace_dir)

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

    lb_idxes = get_lb_timesteps(tr)
    # raveled mats
    all_rmats = read_regr_mats(tr, all_labels)
    all_rmats_nolb = read_regr_mats(tr, all_labels, idx_drop=lb_idxes + 1)

    def run(mat_idx):
        selected_mats = [all_rmats[i] for i in mat_idx]
        X = np.array(selected_mats[:-1])
        y = selected_mats[-1]

        run_regr_actual(X.transpose(), y)

    def run_nolb(mat_idx):
        selected_mats = [all_rmats_nolb[i] for i in mat_idx]
        X = np.array(selected_mats[:-1])
        y = selected_mats[-1]

        run_regr_actual(X.transpose(), y)

    def meanvar(mat):
        mat_var = np.std(mat, axis=1)
        print(np.mean(mat_var))

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
    run_nolb(mat_idx)


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
