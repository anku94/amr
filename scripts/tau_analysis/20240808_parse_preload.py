import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import glob
import numpy as np
import os
import pandas as pd
import re

from datetime import datetime
from dataclasses import dataclass
from typing import TypedDict

from io import StringIO
from common import PlotSaver, plot_init_big
from tabulate import tabulate
from trace_common import TraceSuite, TraceUtils
from plot_types import SingleSpec, BarPlotSpec, plot_bar_plot_spec

trace_dir_prefix = "/mnt/ltio/parthenon-topo"
trace_dir_prefix = "/mnt/ltio/parthenon-topo/mpiallreddbg"


@dataclass
class RunSuite:
    nranks: int
    run_id: str
    trace_names: list[str]
    log_files: list[str]
    amrmon_files: list[str]

    def __str__(self):
        repr_str = f"RunSuite(nranks={self.nranks}, run_id={self.run_id})"
        runs = [f"{t}: {f}" for t, f in zip(self.trace_names, self.log_files)]
        for r in runs:
            repr_str += f"\n- {r}"

        return repr_str


class ParsedRunSuite(TypedDict):
    nranks: int
    run_id: str
    trace_names: list[str]
    section_prof: list[pd.DataFrame]
    section_comm: list[pd.DataFrame]


policies_hum_map = {
    "baseline": "Baseline",
    "cdp": "CDP",
    "cdpc512": "CDP (C=512)",
    "cdpc512par8": "CDP (C=512, P=8)",
    "hybrid25_old": "Hybrid (25%, Old)",
    "hybrid25": "Hybrid (25%)",
    "hybrid50_old": "Hybrid (50%, Old)",
    "hybrid50": "Hybrid (50%)",
    "hybrid75_old": "Hybrid (75%, Old)",
    "hybrid75": "Hybrid (75%)",
    "lpt": "LPT",
}


def get_today() -> str:
    return datetime.now().strftime("%Y%m%d")


def parse_comm_section(section: list[str]) -> pd.DataFrame:
    section = [
        l.strip("\n")
        for l in section
        if l.strip() != ""
        and not l.strip().startswith("----")
        and not l.strip().startswith("I2024")
    ]

    data = StringIO("\n".join(section))
    df = pd.read_fwf(data)
    df.columns = list(map(lambda x: x.strip(":"), df.columns))

    return df


def parse_log_section(section: list[str]) -> pd.DataFrame:
    section = [
        l.strip("\n")
        for l in section
        if l.strip() != ""
        and not l.strip().startswith("----")
        and not l.strip().startswith("I2024")
    ]
    data = StringIO("\n".join(section))
    df = pd.read_fwf(data)
    df.columns = list(map(lambda x: x.strip(":"), df.columns))
    for c in df.columns:
        # if c is a series of type str
        if df[c].dtype == "object":
            df[c] = df[c].str.strip(":")
    return df


def parse_log(log_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(log_file, "r") as f:
        lines = f.readlines()

    dashes = [lidx for (lidx, l) in enumerate(lines) if l.startswith("----")]

    log_prof = lines[dashes[0] - 1 : dashes[1] - 1]
    section_prof = parse_log_section(log_prof)

    log_comm = lines[dashes[-1] - 1 :]
    section_comm = parse_comm_section(log_comm)

    section_prof = section_prof.dropna(axis=0, how="any")

    return section_prof, section_comm


def parse_suite(suite: RunSuite) -> ParsedRunSuite:
    raise NotImplementedError("parse_suite deprecated: use parse_trace_suite")

    parsed_logs = []
    for log_file in suite.log_files:
        parsed_logs.append(parse_log(log_file))

    section_prof, section_comm = zip(*parsed_logs)
    parsed_suite = ParsedRunSuite(
        nranks=suite.nranks,
        run_id=suite.run_id,
        trace_names=suite.trace_names,
        section_prof=section_prof,
        section_comm=section_comm,
    )

    return parsed_suite


def parse_trace_suite(suite: TraceSuite) -> ParsedRunSuite:
    parsed_logs = []
    for log_file in suite.log_files():
        parsed_logs.append(parse_log(log_file))

    section_prof, section_comm = zip(*parsed_logs)
    parsed_suite = ParsedRunSuite(
        nranks=suite.nranks,
        run_id=suite.suite_id,
        trace_names=suite.trace_names(),
        section_prof=section_prof,
        section_comm=section_comm,
    )

    return parsed_suite


def show_prof(parsed_suite: ParsedRunSuite) -> None:
    section_prof = parsed_suite["section_prof"]
    nranks = parsed_suite["nranks"]
    all_df_comp = []

    for i, df in enumerate(section_prof):
        trace_name = parsed_suite["trace_names"][i]
        print(f"------ Trace: {trace_name} ------")

        df["Total"] = df["Count"] * df["Avg"] / (nranks * 1e6)
        df["Total"] = df["Total"].astype(int)
        df.sort_values(by="Total", ascending=False, inplace=True)
        print(df[["Metric", "Total"]].head(20))
        df_comp = df[["Metric", "Total"]]
        df_comp.columns = ["metric", trace_name]

        all_df_comp.append(df_comp)

    df_base = all_df_comp[0]
    df_rest = all_df_comp[1:]

    for df_right in df_rest:
        df_base = df_base.merge(df_right, on="metric", how="outer")

    df_base.sort_values(by="baseline", ascending=False, inplace=True)

    print(df_base.to_string())


def show_prof_detailed(parsed_suite: ParsedRunSuite) -> None:
    section_prof = parsed_suite["section_prof"]
    nranks = parsed_suite["nranks"]
    all_df_comp = []

    for i, df in enumerate(section_prof):
        trace_name = parsed_suite["trace_names"][i]
        trace_name = re.sub(r"hybrid", "h", trace_name)  # for brevity
        print(f"------ Trace: {trace_name} ------")

        df_comp = pd.DataFrame()
        df_comp["metric"] = df["Metric"]
        df_comp[f"{trace_name}_tot_s"] = (
            df["Count"] * df["Avg"] / (nranks * 1e6)
        ).astype(int)
        df_comp["count"] = df["Count"]
        df_comp["min_ms"] = (df["Min"] / 1e3).astype(int)
        df_comp["max_ms"] = (df["Max"] / 1e3).astype(int)
        df_comp["avg_ms"] = (df["Avg"] / 1e3).astype(int)
        df_comp.sort_values(by=f"{trace_name}_tot_s", ascending=False, inplace=True)

        # print(df_comp.head(20))
        all_df_comp.append(df_comp)

    df_base = all_df_comp[0]
    df_rest = all_df_comp[1:]

    col_to_sort = df_base.columns[1]

    for df_right in df_rest:
        df_base = df_base.merge(df_right, on="metric", how="outer")

    df_base.sort_values(by=col_to_sort, ascending=False, inplace=True)

    print(df_base.to_string())
    # print(df_base.to_markdown())
    # print(tabulate(df_base, headers="keys", tablefmt="grid"))


def get_bar_plot_spec(trace_suite: TraceSuite) -> BarPlotSpec:
    parsed_suite = parse_trace_suite(trace_suite)
    all_funcs = [
        ("Driver_Main", "total"),
        ("ConToPrim::Solve", "c2p"),
        ("CalculateFluxes", "calcflx"),
        ("UpdateMeshBlockTree", "umbt"),
        ("RedistributeAndRefineMeshBlocks", "rnrmb"),
        ("MPI_Allgather", "mpiag"),
        ("MPI_Allreduce", "mpiar"),
        # ("MPI_Barrier", "mpibar"),
        # ("Step 7", "s7"),
        # ("Step 8", "s8"),
        # ("Step 9", "s9"),
        ("Mesh::Initialize", "meshinit"),
    ]

    def get_func_data(func: str):
        func_data = []
        for df in parsed_suite["section_prof"]:
            func_row = df[df["Metric"].str.contains(func)].iloc[0]
            func_time_us = func_row["Count"] * func_row["Avg"] / (trace_suite.nranks)
            func_data.append(func_time_us)

        return np.array(func_data)

    bpspec: BarPlotSpec = {
        "trace_key": trace_suite.suite_id,
        "trace_names": trace_suite.trace_names(),
        "nranks": trace_suite.nranks,
        "keys": [f[1] for f in all_funcs],
        "data": {f[1]: get_func_data(f[0]) for f in all_funcs},
        "plot_fname": "bar_libprof",
    }

    return bpspec


def run_blast_wave():
    TraceUtils.desc_traces()

    trace_suite_key = "cdppar2048"
    trace_suite_key = "cdppar4096.first"
    trace_suite_key = "mpidbgwbar"
    trace_suite_key = "mpidbg4096ub22.12"
    trace_suite_key = "mpidbg4096ub22.18"
    trace_suite = TraceUtils.get_traces(trace_suite_key)
    trace_suite

    print(trace_suite)
    # parsed_suite = parse_trace_suite(trace_suite)
    # show_prof_detailed(parsed_suite)

    bpspec = get_bar_plot_spec(trace_suite)
    bpspec
    plot_bar_plot_spec(bpspec)
    return

    prof_phases = ["app", "comp", "comm", "sync", "lb"]

    prof_data = [
        classify_prof_phases(trace_suite.nranks, df)
        for df in parsed_suite["section_prof"]
    ]

    print(prof_data)
    plot_prof(parsed_suite, prof_phases, prof_data)

    # parsed_suite = parse_trace_suite(trace_suite)

    return

    log_files[1]
    log_file = log_files[1]

    prof0 = parsed_suite["section_prof"][0]
    prof1 = parsed_suite["section_prof"][1]
    prof1.columns
    df = prof0
    # get type of each df column
    for c in df.columns:
        print(f"{c}: {df[c].dtype}")

    classify_prof_phases(nranks, parsed_suite["section_prof"][0])
    classify_prof_phases(nranks, parsed_suite["section_prof"][1])

    prof_data = [
        classify_prof_phases(nranks, df) for df in parsed_suite["section_prof"]
    ]

    print(prof_data)
    plot_prof(parsed_suite, prof_phases, prof_data)


def run():
    plot_init_big()
    # run_parse()
    # run_for_mochi()
    run_blast_wave()


if __name__ == "__main__":
    run()
