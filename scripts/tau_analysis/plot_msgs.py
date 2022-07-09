import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pickle
import IPython


def plot_init():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def to_dense(idx, cnt, nranks=512):
    dense_arr = []

    csrint = lambda x: [int(i) for i in x.split(",")]
    keys = csrint(idx)
    vals = csrint(cnt)
    sdict = dict(zip(keys, vals))

    for i in range(nranks):
        i_val = 0
        if i in sdict:
            i_val = sdict[i]
        dense_arr.append(i_val)

    return np.array(dense_arr)


def to_dense_2d(df, nts, nranks=512):
    all_rows = {}

    for idx, row in df.iterrows():
        ts = row["timestep"]
        idxes = row["rank"]
        counts = row["msg_sz_count"]
        row_dense = to_dense(idxes, counts, nranks)

        if ts in all_rows:
            all_rows[ts] += row_dense
        else:
            all_rows[ts] = row_dense

    all_rows_dense = []

    for ts in range(nts):
        ts_row = np.zeros(nranks, dtype=int)
        if ts in all_rows:
            ts_row = all_rows[ts]

        all_rows_dense.append(ts_row)

    mat_2d = np.stack(all_rows_dense, axis=0)
    return mat_2d


def sort_mat_by_rank(mat_2d):
    rows = mat_2d.tolist()

    append_idx = lambda x: list(enumerate(x))
    sort2 = lambda x: sorted(x, key=lambda x: x[1])
    strip2 = lambda x: [i[0] for i in x]
    sorted_idx = lambda x: strip2(sort2(append_idx(x)))

    all_rank_rows = []

    for row in rows:
        rank_row = sorted_idx(row)
        all_rank_rows.append(rank_row)

    all_rank_mat = np.stack(all_rank_rows, axis=0)
    print(all_rank_mat.shape)
    print(all_rank_mat[0][0])
    return all_rank_mat


def get_rankgrid(df_msg, labels, send_or_recv=9):
    nsteps = df_msg['timestep'].max()
    nranks = 512

    if send_or_recv in [0, 1]:
        df_msg = df_msg[df_msg['send_or_recv'] == send_or_recv]

    mat_2d = np.zeros((nsteps, nranks), dtype=int)

    for label in labels:
        df_cur = df_msg[df_msg["phase"] == label]
        mat_2d_cur = to_dense_2d(df_msg, nsteps, nranks)
        mat_2d += mat_2d_cur

    mat_ranks = sort_mat_by_rank(mat_2d)
    print(mat_ranks.shape)
    return mat_ranks


def plot_msg_rankgrid(rankgrid, labels, send_or_recv, plot_dir):
    print(rankgrid)

    fig, ax = plt.subplots(1, 1)

    ax_im = ax
    im = ax_im.imshow(rankgrid, aspect="auto", cmap="plasma")

    fig.colorbar(im, ax=ax)
    ax_im.xaxis.set_ticks([])

    title = '_'.join(labels)
    if send_or_recv not in [0, 1]:
        send_or_recv = 2

    title_suffix = {
        0: 'Send',
        1: 'Recv',
        2: 'Send+Recv'
    }[send_or_recv]

    ax_im.set_title('{} ({})'.format(title, title_suffix))
    ax_im.set_ylabel('Timesteps (increasing downward)')

    save = True
    save_path = '{}/{}_{}.pdf'.format(plot_dir, '_'.join(labels), send_or_recv)
    if save:
        fig.savefig(save_path, dpi=600)
    else:
        fig.show()


def run_plot_msg_rankgrid(df_msg):
    lab_lb = "LoadBalancing"
    lab_fx = "FluxExchange"
    lab_bc = "BoundaryComm"

    plot_dir = 'figures/messages'

    labels = [lab_lb]
    mat = get_rankgrid(df_msg, labels)
    plot_msg_rankgrid(mat, labels, 3, plot_dir)

    labels = [lab_fx, lab_bc]
    mat = get_rankgrid(df_msg, labels)
    plot_msg_rankgrid(mat, labels, 3, plot_dir)


def run():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    df_msg_path = "{}/aggr/msg_concat.csv".format(trace_dir)
    df_msg = pd.read_csv(df_msg_path)
    df_msg = df_msg.astype(
        {
            "timestep": int,
            "phase": str,
            "send_or_recv": int,
            "rank": str,
            "msg_sz_count": str,
        }
    )

    run_plot_msg_rankgrid(df_msg)


if __name__ == "__main__":
    run()
