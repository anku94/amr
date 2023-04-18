import glob
import multiprocessing
import numpy as np
import pandas as pd
import ipdb
import pickle
import subprocess
import sys
import time

import ray
import traceback
from typing import Tuple

from pathlib import Path
from task import Task
from trace_reader import TraceOps
from typing import List

prev_output_ts = 0

phase_boundaries = None


def joinstr(x):
    return ",".join([str(i) for i in x])


def joinstr2(x):
    return "|".join([str(i) for i in x])


def find_func_remove_mismatch(ts_begin, ts_end):
    all_inv = []

    for ts in ts_begin:
        all_inv.append((ts, 0))

    for ts in ts_end:
        all_inv.append((ts, 1))

    all_inv = sorted(all_inv)

    all_begin_clean = []
    all_end_clean = []

    prev_type = 1
    prev_ts = None

    for ts, ts_type in all_inv:
        assert prev_type in [0, 1]
        assert ts_type in [0, 1]

        if ts_type == 0:
            prev_type = 0
            prev_ts = ts

        if prev_type == 0 and ts_type == 1:
            all_begin_clean.append(prev_ts)
            all_end_clean.append(ts)
            prev_type = 1
            prev_ts = None

    return all_begin_clean, all_end_clean


def find_func(df, func_name):
    #  print('finding {}'.format(func_name))

    ts_begin = df[(df["func"] == func_name) & (df["enter_or_exit"] == 0)]["timestamp"]
    ts_end = df[(df["func"] == func_name) & (df["enter_or_exit"] == 1)]["timestamp"]
    #  print(ts_begin)
    #  print(ts_end)

    all_invocations = []

    try:
        assert ts_begin.size == ts_end.size
        all_invocations = list(zip(ts_begin.array, ts_end.array))
    except AssertionError as e:
        print(func_name, ts_begin.size, ts_end.size)
        print(traceback.format_exc())
        ts_begin, ts_end = find_func_remove_mismatch(ts_begin.array, ts_end.array)
        all_invocations = list(zip(ts_begin, ts_end))

    #  print(func_name)
    #  print(all_invocations)

    return all_invocations


def add_to_ts_beg(all_phases, phase_ts, phase_label):
    for pbeg, _ in phase_ts:
        all_phases.append((pbeg, phase_label + ".BEGIN"))


def add_to_ts_end(all_phases, phase_ts, phase_label):
    for _, pend in phase_ts:
        all_phases.append((pend, phase_label + ".END"))


"""
Multiple AR2.BEGINs can appear simultaneously
Multiple AR2.ENDs can appear intermittently (FillDerived)
All regions within AR3 are to be ignored, except AR3_ class

What this does:

* Multiple BEGINs for same thing: accept first BEGIN
* AR1.END - ignore. Multiple AR1.ENDs possible - not a reliable marker.
* AR2.END - ignore. Multiple AR2.ENDs possible - not a reliable marker.
* SR.END - ignore. Multiple SR.ENDs possible - not a reliable marker.
* Any BEGIN - END prev active frame on the same level
"""


def filter_and_add_missing(phases):
    cur_stack = []
    cur_stack_ts = []
    filtered_phases = []

    for phase_ts, phase_key in phases:
        phase_name, beg_or_end = phase_key.split(".")

        if "AR3" in cur_stack and not phase_name.startswith("AR3"):
            continue

        stack_depth = len(cur_stack)
        phase_depth = len([i for i in phase_name if "_" in i])

        if beg_or_end == "BEGIN":
            #  print(f'BEGIN {phase_name}, {cur_stack} (phase_depth: {phase_depth})')
            # Ignore duplicate BEGINs
            if phase_name in cur_stack:
                continue

            # END all BEGINs at higher nesting
            while len(cur_stack) > phase_depth:
                active_top = cur_stack.pop()
                active_top_ts = cur_stack_ts.pop()
                filtered_phases.append((phase_ts, active_top + ".END"))
                #  print(f'Closing phase {active_top}')

            cur_stack.append(phase_name)
            cur_stack_ts.append(phase_ts)
            filtered_phases.append((phase_ts, phase_key))
        elif beg_or_end == "END":
            if phase_name in ["AR1", "AR2", "SR"]:
                continue

            if not cur_stack:
                ipdb.set_trace()

            last_phase = cur_stack.pop()
            cur_stack_ts.pop()
            assert last_phase == phase_name
            filtered_phases.append((phase_ts, phase_key))

    assert len(cur_stack) == 0

    return filtered_phases


def validate_phases(phases):
    #  phases = sorted(phases)

    cur_stack = []

    for phase_ts, phase_name in phases:
        cur_begin = False
        cur_name = phase_name.split(".")[0]

        if phase_name.endswith(".BEGIN"):
            cur_begin = True
        elif phase_name.endswith(".END"):
            cur_begin = False
        else:
            assert False

        if cur_begin:
            cur_stack.append(cur_name)
        else:
            assert len(cur_stack) > 0
            cur_open_name = cur_stack.pop()
            assert cur_open_name == cur_name

    assert len(cur_stack) == 0


def classify_phases(df_ts) -> List:
    phases = []

    global phase_boundaries

    phase_boundaries = {
        "AR1": ["Reconstruct", "Task_ReceiveFluxCorrection"],
        "AR1_SEND": ["Task_SendFluxCorrection", "Task_SendFluxCorrection"],
        "AR1_RECV": ["Task_ReceiveFluxCorrection", "Task_ReceiveFluxCorrection"],
        "AR2": ["Task_ClearBoundary", "Task_FillDerived"],
        "AR2_FD": ["Task_FillDerived", "Task_FillDerived"],
        "AR3": [
            "LoadBalancingAndAdaptiveMeshRefinement",
            "LoadBalancingAndAdaptiveMeshRefinement",
        ],
        "AR3_UMBT": ["UpdateMeshBlockTree", "UpdateMeshBlockTree"],
        "SR": ["Task_SendBoundaryBuffers_MeshData", "Task_SetBoundaries_MeshData"],
        "SR_SEND": [
            "Task_SendBoundaryBuffers_MeshData",
            "Task_SendBoundaryBuffers_MeshData",
        ],
        "SR_SET": ["Task_SetBoundaries_MeshData", "Task_SetBoundaries_MeshData"],
    }

    for phase, bounds in phase_boundaries.items():
        if bounds[0] is not None:
            ret = find_func(df_ts, bounds[0])
            add_to_ts_beg(phases, ret, phase)
        if bounds[1] is not None:
            ret = find_func(df_ts, bounds[1])
            add_to_ts_end(phases, ret, phase)

    def phase_sort_key(k):
        kt = [0, 0]
        kt[0] = k[0]
        kt[1] = 0 if "_" in k[1] else 1

        pn = k[1]
        if pn.endswith(".BEGIN"):
            kt[1] = 1 if "_" in pn else 0
        elif pn.endswith(".END"):
            kt[1] = 0 if "_" in pn else 1

        return kt

    phases = sorted(phases, key=phase_sort_key)

    def print_phases(phases, sep1, sep2):
        print(sep1 * 20)
        for phase in phases:
            print(phase)
        print(sep2 * 20)

    #  print_phases(phases, '=', '-')

    try:
        phases_cleaned = filter_and_add_missing(phases)
    except Exception as e:
        #  ipdb.set_trace()
        print("[Error] Exception in filter_and_add_missing")
        raise Exception("failure: filter_aam")

    #  print_phases(phases_cleaned, '-', '=')

    try:
        validate_phases(phases_cleaned)
    except AssertionError as e:
        print("[Error] Exception in validate_phases")
        raise Exception("failure: validate_phases")

    return phases_cleaned


def aggregate_phases(df, phases):
    phase_total = {}
    cur_phase = None
    cur_phase_begin = -1

    def add_phase(phase, phase_time):
        if phase in phase_total:
            phase_total[phase] += phase_time
        else:
            phase_total[phase] = phase_time

    active_phases = {}

    for phase_ts, phase in phases:
        phase_name = phase.split(".")[0]

        if phase.endswith(".BEGIN"):
            active_phases[phase_name] = phase_ts
        elif phase.endswith(".END"):
            if phase_name in active_phases:
                phase_time = phase_ts - active_phases[phase_name]
                del active_phases[phase_name]
                add_phase(phase_name, phase_time)

    total_phasewise = 0
    all_phases = phase_boundaries.keys()

    for key in all_phases:
        if key in phase_total:
            total_phasewise += phase_total[key]

    total_ts = df["timestamp"].max() - df["timestamp"].min()

    #  print('Phases: {}, Total: {}, Accounted: {:.0f}%'.format(total_phasewise, total_ts, total_phasewise * 100.0 / total_ts))

    return phase_total, total_phasewise, total_ts


def log_event(f, rank, ts, evt_name, evt_val):
    f.write("{:d},{:d},{},{:d}\n".format(rank, ts, evt_name, evt_val))


def process_df_for_ts(rank, ts, df_ts, f):
    df_ts = df_ts[df_ts["enter_or_exit"].isin([0, 1])]
    df_ts = df_ts.dropna().astype(
        {
            "rank": "int32",
            "timestep": "int64",
            "timestamp": "int64",
            "func": str,
            "enter_or_exit": int,
        }
    )

    #  phases, validation_passed = classify_phases(df_ts)
    try:
        phases = classify_phases(df_ts)
    except Exception as e:
        print(f"[Error] Rank {rank}: classify phases failed for ts: {ts}")
        raise Exception("failure: classify_phases")

    phase_total, total_phasewise, total_ts = aggregate_phases(df_ts, phases)

    global prev_output_ts

    output_call = find_func(df_ts, "MakeOutputs")
    cur_output_ts = output_call[0][1]

    for phase, phase_time in phase_total.items():
        log_event(f, rank, ts, phase, phase_time)

    total_ts_fromprev = cur_output_ts - prev_output_ts

    # for the first time
    if prev_output_ts == 0:
        total_ts_fromprev = total_ts

    log_event(f, rank, ts, "TIME_CLASSIFIEDPHASES", total_phasewise)
    log_event(f, rank, ts, "TIME_FROMCURBEGIN", total_ts)
    log_event(f, rank, ts, "TIME_FROMPREVEND", total_ts_fromprev)

    prev_output_ts = cur_output_ts


"""
classify_trace:

CAN NOT raise an exception. Should be handled internally.
"""


def classify_trace(rank, in_path, out_path):
    df = pd.read_csv(
        in_path, sep="|", usecols=range(5), lineterminator="\n", low_memory=False
    )

    df = df.dropna().astype(
        {
            "rank": "int32",
            "timestep": "int64",
            "timestamp": "int64",
            "func": str,
            "enter_or_exit": int,
        }
    )

    df["group"] = np.where(
        (df["func"] == "MakeOutputs") & (df["enter_or_exit"] == 1), 1, 0
    )
    df["group"] = df["group"].shift(1).fillna(0).astype(int)
    df["group"] = df["group"].cumsum()
    all_dfs = df.groupby("group", as_index=False)

    with open(out_path, "w+") as f:
        header = "rank,ts,evtname,evtval\n"
        f.write(header)
        for ts, df_ts in all_dfs:
            #  if ts < 26: continue
            #  print(rank, ts)
            #  print(df_ts.to_string())
            #  df_ts = all_dfs.get_group(3807)
            try:
                process_df_for_ts(rank, ts, df_ts, f)
            except Exception as e:
                print(rank, ts)
                #  print(df_ts.to_string())
                print(e)
                print(traceback.format_exc())
                #  sys.exit(-1)
            #  if ts > 25: break
            #  break


def read_classified_trace(args):
    trace_out = args["out_path"]
    df = pd.read_csv(trace_out).astype(
        {"rank": "int32", "ts": "int32", "evtname": str, "evtval": "int64"}
    )
    df["rank"] = args["rank"]
    return df


def classify_trace_worker(args):
    trace_in = args["in_path"]
    trace_out = args["out_path"]
    rank = args["rank"]
    print("Parsing {} into {}...".format(trace_in, trace_out))
    classify_trace(rank, trace_in, trace_out)


@ray.remote
def classify_trace_parworker(args):
    classify_trace_worker(args)


class TraceClassifier(Task):
    def __init__(self, trace_dir):
        super().__init__(trace_dir)

    def gen_worker_fn_args_rank(self, rank):
        in_path = "{}/trace/funcs/funcs.{}.csv".format(self._trace_dir, rank)
        out_path = "{}/phases/phases.{}.csv".format(self._trace_dir, rank)

        args = super().gen_worker_fn_args_rank(rank)
        args["in_path"] = in_path
        args["out_path"] = out_path

        return args

    @staticmethod
    def worker(args):
        ret = classify_trace_worker(args)
        return ret


class ParallelReader(TraceClassifier):
    def __init__(self, trace_dir):
        super().__init__(trace_dir)

    @staticmethod
    def worker(args):
        ret = read_classified_trace(args)
        return ret


def run_classify(trace_dir):
    phase_path = Path(trace_dir) / "phases"
    phase_path.mkdir(parents=True, exist_ok=True)
    tc = TraceClassifier(trace_dir)
    #  tc.run_rank(467)
    #  tc.run_node()
    ray.init(address="h0:6379")
    tc.run_func_with_ray(classify_trace_parworker)

    return


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = "percentile_%s" % n
    return percentile_


def run_parse_log(dpath: str):
    log_path = "{}/run/log.txt".format(dpath)
    df_path = "{}/trace/logstats.csv".format(dpath)

    print("Analyzing {}".format(log_path))
    print("Writing {}".format(df_path))

    f = open(log_path).read().split("\n")
    data = [line for line in f if "zone-cycles/wsec" in line]

    keys = "cycle,time,dt,zc_per_step,wtime_total,wtime_step_other,zc_wamr,wtime_step_amr".split(
        ","
    )
    vals = [
        [float(i.split("=")[1]) for i in data[k].split(" ")] for k in range(len(data))
    ]


    df = pd.DataFrame.from_records(vals, columns=keys).astype({"cycle": int})
    df.to_csv(df_path, index=None)


""" combine runtime values for each timestep + event """


def run_aggregate(trace_dir):
    pr = ParallelReader(trace_dir)
    all_phase_dfs = pr.run_rankwise(0, 512)
    phase_df = pd.concat(all_phase_dfs).dropna()

    # aggr1
    aggr_df1_path = "{}/trace/phases.aggr.csv".format(trace_dir)
    print("Writing {}".format(aggr_df1_path))

    aggr_df1 = phase_df.groupby(["evtname", "rank"], as_index=False).agg(
        {"evtval": "sum"}
    )

    aggr_df1 = (
        aggr_df1.sort_values(["evtname", "rank"])
        .groupby("evtname", as_index=False)
        .agg({"evtval": joinstr, "rank": joinstr})
    )

    aggr_df1.to_csv(aggr_df1_path, index=None)

    # aggr2
    aggr_df2_path = "{}/trace/phases.aggr.by_ts.csv".format(trace_dir)
    print("Writing {}".format(aggr_df2_path))

    aggr_df2 = (
        phase_df.sort_values(["ts", "evtname", "rank"])
        .groupby(["ts", "evtname"], as_index=False)
        .agg({"evtval": list})
    )

    aggr_df2.to_csv(aggr_df2_path, index=None)


if __name__ == "__main__":
    """Notes on exception handling:
    All exceptions were made non-fatal on Feb 20, 2023.
    This is to handle cases where the profiler doesn't fully emit the last timestep
    (for various reasons)
    In case an analysis is failing at some intermediate timestep,
    this should be re-enabled."""

    trace_dir = "/mnt/ltio/parthenon-topo/profile17"
    #  run_classify(trace_dir)
    # XXX: run_parse_log probably not necessary
    # generates logstats.csv, but log.txt.csv already generated by phoebus_runs.sh
    run_parse_log(trace_dir)
    #  run_aggregate(trace_dir)
