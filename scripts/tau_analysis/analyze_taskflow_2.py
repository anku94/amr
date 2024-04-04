import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

from analyze_taskflow import find_func

from trace_reader import TraceOps
from task import Task

ray.init(address="h0:6379")


def missing_gaps(df):
    df_mo = df[(df["func"] == "MakeOutputs") & (df["enter_or_exit"] == 0)]

    diff_total = 0

    for row_idx, row in df_mo.iterrows():
        prev_row = df.loc[row_idx - 1]
        prev_ts = prev_row["timestamp"]
        cur_ts = row["timestamp"]
        diff_ts = cur_ts - prev_ts
        diff_total += diff_ts

    return diff_total


def compute_fdtime_inlb(df):
    df_lb = df[df["func"] == "LoadBalancingAndAdaptiveMeshRefinement"]
    df_lb = df_lb.astype({"enter_or_exit": int}).copy()

    df_lb_op = df_lb[df_lb["enter_or_exit"] == 0]
    df_lb_cl = df_lb[df_lb["enter_or_exit"] == 1]
    idx_op = df_lb_op.index.tolist()
    idx_cl = df_lb_cl.index.tolist()
    lb_pair = list(zip(idx_op, idx_cl))

    fd_time_tot = 0

    for lb in lb_pair:
        df_lb = df.iloc[lb[0] : lb[1] + 1]
        fd_time = find_func(df_lb, "Task_FillDerived")
        for fd_pair in fd_time:
            pair_diff = fd_pair[1] - fd_pair[0]
            fd_time_tot += pair_diff

    return fd_time_tot


@ray.remote
def analyze_fd(args):
    rank = args["rank"]

    tr = TraceOps(trace_dir)
    df = tr.trace.read_rank_trace(rank).dropna().astype({"enter_or_exit": int})

    time_gap = missing_gaps(df)
    time_fd = compute_fdtime_inlb(df)

    return time_gap, time_fd


class MissingGapCalc(Task):
    def __init__(self, trace_dir):
        super().__init__(trace_dir)

    @staticmethod
    def worker(fn_args):
        return find_missing_gaps(fn_args)

    """ Gap between the function Task_EstimateTimestep and MakeOutputs
    in AR3_LB region. Presumably hides a barrier that some ranks spend
    time waiting at. This gap is inversely proportional to the time
    spent in FillDerived calls in AR3_LB. This plot visually confirms that.
    """

    @staticmethod
    def plot_gap_vs_fd(data, plot_dir):
        fig, ax = plt.subplots(1, 1)

        data = list(zip(*data))
        data_gap = np.array(data[0])
        data_fd = np.array(data[1])
        data_x = range(len(data_gap))

        ax.plot(data_x, data_gap, label="Gap:ETS-MO")
        ax.plot(data_x, data_fd, label="Time:T_FD")

        ax.plot(data_x, data_gap + data_fd, label="Sum Of Both")

        ax.legend()

        ax.set_xlabel("Rank")
        ax.set_ylabel("Time")
        ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f} s".format(x / 1e6))

        plot_path = "{}/gap_vs_fd.pdf".format(plot_dir)
        fig.savefig(plot_path)


def run_missing_gaps():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    mgc = MissingGapCalc(trace_dir)
    #  ret = mgc.run_rankwise(0, 512)
    data = mgc.run_rank(0)
    data = mgc.run_func_with_ray(analyze_fd)
    plot_dir = "figures/other"
    mgc.plot_gap_vs_fd(data, plot_dir)


def run():
    run_missing_gaps()


if __name__ == "__main__":
    run()
