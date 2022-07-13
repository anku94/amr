import collections

import colorcet
import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

from trace_reader import TraceReader


class TraceOps:
    trace = None

    @classmethod
    def split_eqn(cls, eqn):
        eqn = "a+b-c-d-e+f"
        symbols = re.split("\+|-", eqn)

        def sign(eqn, symbol):
            idx = eqn.find(symbol)
            symbol = "+"
            if idx > 0:
                symbol = eqn[idx - 1]
            return symbol

        pos = [s for s in symbols if sign(eqn, s) == "+"]
        neg = [s for s in symbols if sign(eqn, s) == "-"]
        return [pos, neg]

    @classmethod
    def cropsum_2d(cls, mats):
        len_min = min([m.shape[0] for m in mats])
        mats = [m[:len_min] for m in mats]
        print([m.shape for m in mats])
        mat_agg = np.sum(mats, axis=0)
        print(mat_agg.shape)
        return mat_agg

    @classmethod
    def multimat_labels(cls, labels, f):
        labels = cls.split_eqn(labels)
        labels_pos = labels[0]
        labels_neg = labels[1]

        mats_pos = [f(l) for l in labels_pos]
        mat_pos_agg = cls.cropsum_2d(mat_pos)

        if len(labels_neg) > 0:
            mats_neg = [f(l) for l in labels_neg]
            mat_neg_agg = cls.cropsum_2d(mat_neg)

            return mat_pos_agg - mat_neg_agg

        return mat_pos_agg

    @classmethod
    def multimat_tau(cls, labels):
        labels = labels.split("+")
        mats = [cls.trace.get_tau_event(l) for l in labels]
        mat_agg = cls.cropsum_2d(mats)
        return mat_agg

    @classmethod
    def multimat_msg(cls, labels):
        labels = labels.split("+")
        mats = [cls.trace.get_msg_count(l) for l in labels]
        mat_agg = cls.cropsum_2d(mats)
        return mat_agg

    @classmethod
    def multimat(cls, label_str):
        ltype, labels = label_str.split(":")
        if ltype == "tau":
            return cls.multimat_labels(labels, cls.trace.get_tau_event)
        elif ltype == "msgcnt":
            return cls.multimat_labels(labels, cls.trace.get_msg_count)
        elif ltype == "npeer":
            return cls.multimat_labels(labels, cls.trace.get_msg_npeers)
        else:
            assert False


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

    def cropsum_2d(mats):
        len_min = min([m.shape[0] for m in mats])
        mats = [m[:len_min] for m in mats]
        print([m.shape for m in mats])
        mat_agg = np.sum(mats, axis=0)
        print(mat_agg.shape)
        return mat_agg

    def multimat_labels(labels, f):
        labels = labels.split("+")
        mats = [f(l) for l in labels]
        mat_agg = cropsum_2d(mats)
        return mat_agg

    def multimat_tau(labels):
        labels = labels.split("+")
        mats = [tr.get_tau_event(l) for l in labels]
        mat_agg = cropsum_2d(mats)
        return mat_agg

    def multimat_msg(labels):
        labels = labels.split("+")
        mats = [tr.get_msg_count(l) for l in labels]
        mat_agg = cropsum_2d(mats)
        return mat_agg
        pass

    def multimat(label_str):
        ltype, labels = label_str.split(":")
        if ltype == "tau":
            return multimat_labels(labels, tr.get_tau_event)
        elif ltype == "msgcnt":
            return multimat_labels(labels, tr.get_msg_count)
        elif ltype == "npeer":
            return multimat_labels(labels, tr.get_msg_npeers)
        else:
            assert False

    all_evts = ["AR1", "AR2", "AR3", "AR3_UMBT", "SR"]

    evty_label = "tau:AR1+AR2+SR"
    evty_label = "tau:SR"
    evty_mat = multimat(evty_label)
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


def run_regression():
    pass


if __name__ == "__main__":
    run_plot()
