import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import glob
import numpy as np
import pandas as pd

from datetime import datetime
from typing import TypedDict

from io import StringIO
from common import PlotSaver, plot_init_big


class RunSuite(TypedDict):
    nranks: int
    trace_names: list[str]
    log_files: list[str]


class ParsedRunSuite(TypedDict):
    nranks: int
    trace_names: list[str]
    section_prof: list[pd.DataFrame]
    section_comm: list[pd.DataFrame]


def get_today() -> str:
    return datetime.now().strftime("%Y%m%d")


def parse_log_section(section: list[str]) -> pd.DataFrame:
    section = [l.strip('\n') for l in section if l.strip() !=
               "" and not l.strip().startswith("----")]
    data = StringIO('\n'.join(section))
    df = pd.read_fwf(data)
    df.columns = list(map(lambda x: x.strip(':'), df.columns))
    for c in df.columns:
        # if c is a series of type str
        if df[c].dtype == 'object':
            df[c] = df[c].str.strip(':')
    return df


def parse_log(log_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(log_file, 'r') as f:
        lines = f.readlines()
        pass

    dashes = [lidx for (lidx, l) in enumerate(lines) if l.startswith("----")]
    section_prof = parse_log_section(lines[dashes[0] - 1:dashes[1] - 1])
    section_comm = parse_log_section(lines[dashes[-1] - 1:])

    return section_prof, section_comm


def parse_suite(suite: RunSuite) -> ParsedRunSuite:
    parsed_logs = []
    for log_file in suite["log_files"]:
        parsed_logs.append(parse_log(log_file))

    section_prof, section_comm = zip(*parsed_logs)
    parsed_suite = ParsedRunSuite(
        nranks=suite["nranks"],
        trace_names=suite["trace_names"],
        section_prof=section_prof,
        section_comm=section_comm
    )

    return parsed_suite


def classify_prof_phases(nranks: int, df: pd.DataFrame) -> dict[str, int]:
    metric_map = {
        "Driver_Main": "app",
        "UpdateMeshBlockTree": "sync",
        "RedistributeAndRefineMeshBlocks": "lb",
        "Task_LoadAndSendBoundBufs": "comm",
        "Task_ReceiveBoundBufs": "comm",
        "Task_FillDerived": "comp"}

    phase_times: dict[str, int] = {}

    df_slice: pd.DataFrame = pd.DataFrame(df[['Metric', 'Count', 'Avg']].copy())
    df_slice['Time'] = df_slice['Avg'] * df_slice['Count'] / nranks
    df_slice['Time'] = df_slice['Time'].astype(int)

    print(df_slice)

    for _, row in df_slice.iterrows():
        metric: str = str(row["Metric"])
        time: int = int(row["Time"])

        if metric not in metric_map:
            continue

        phase = metric_map[metric]
        if phase not in phase_times:
            phase_times[phase] = 0

        phase_times[phase] += int(time / 1e6)

    return phase_times


def plot_prof(parsed_suite: ParsedRunSuite, prof_phases: list[str], prof_data: list[dict[str, int]]):
    nranks = parsed_suite["nranks"]
    nphases = len(prof_phases)
    nphases_x = np.arange(nphases)
    traces = parsed_suite["trace_names"]

    ntraces = len(traces)

    width = 1 / (ntraces + 1)


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i, trace in enumerate(traces):
        trace_prof_data = prof_data[i]
        trace_prof_data = [trace_prof_data[p] for p in prof_phases]

        data_x = nphases_x + i * width
        ax.bar(data_x, trace_prof_data, width,
               label=trace, edgecolor="black", zorder=2)

    ax.set_xticks(nphases_x + width * (len(traces) - 1) / 2)
    ax.set_xticklabels(prof_phases)
    ax.legend(title="Traces")

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)

    ymax = max([x["app"] for x in prof_data])
    ylim = int(np.ceil(ymax / 1000) * 1000)
    ax.set_ylim(bottom=0, top=ylim)

    ax.set_xlabel("App Phase")
    ax.set_ylabel("Phase Time (s)")
    ax.set_title(f"Phase-Wise Perf Breakdown (nranks={nranks})")
    fig.tight_layout()

    today = get_today()
    fname = f"{today}_prof_phases_nranks{nranks}"
    PlotSaver.save(fig, "", None, fname)


def convert_size_to_gb(size: str) -> float:
    size = size.strip()
    size_bytes = 0

    if size.endswith("GiB"):
        size_bytes = float(size[:-3]) * 1024 * 1024 * 1024
    elif size.endswith("MiB"):
        size_bytes = float(size[:-3]) * 1024 * 1024
    elif size.endswith("KiB"):
        size_bytes = float(size[:-3]) * 1024
    elif size.endswith("B"):
        size_bytes = float(size[:-1])
    else:
        raise ValueError(f"Invalid size format: {size}")

    size_gb = size_bytes / 2**30
    return size_gb


def plot_comm_mochi(suite: ParsedRunSuite):
    comm_dfs = [df for df in suite["section_comm"]]
    comm_local = [df.iloc[1]['Local'] for df in comm_dfs]
    comm_total = [df.iloc[1]['Global'] for df in comm_dfs]

    num = lambda x: float(x.replace(" GiB", ""))

    comm_local = np.array([num(x) for x in comm_local])
    comm_total = np.array([num(x) for x in comm_total])

    print(comm_local)
    print(comm_total)

    #  comm_local = np.array([convert_size_to_gb(s) for s in comm_local])
    #  comm_total = np.array([convert_size_to_gb(s) for s in comm_total])

    comm_local_pct = comm_local / comm_total
    print(comm_local_pct)

    data_x = np.arange(len(comm_local))

    print(data_x)
    width = 0.3

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(data_x, comm_local_pct, width=width)
    #  ax.bar(data_x, comm_local, width, label="Local",
           #  edgecolor="black", zorder=2)
    #  ax.bar(data_x + width, comm_total, width,
           #  label="Total", edgecolor="black", zorder=2)

    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)

    # ymax = max(comm_total)
    ymax = 1
    # ylim = int(np.ceil(ymax / 100) * 100)
    ax.set_ylim(bottom=0, top=1)

    ax.set_xticks(data_x + width / 2)
    ax.set_xticklabels(suite["trace_names"])
    #ax.legend(title="Comm Type")
# 
    ax.set_xlabel("Trace")
    ax.set_ylabel("Data Exchange (GiB)")
    ax.set_title(f"P2P Comm Volume (nranks={suite['nranks']})")

    nranks = suite["nranks"]

    today = get_today()
    fname = f"{today}_comm_size_nranks{nranks}"
    PlotSaver.save(fig, "", None, fname)


def run_parse_suite():
    nranks = 512
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    traces = ["stochsg52", "stochsg53", "stochsg54", "stochsg55"]
    policies = ["Baseline", "LPT", "Contiguous-DP", "CDPP"]
    log_files = [trace_dir_fmt.format(t) + "/run/log.txt" for t in traces]

    suite = RunSuite(nranks=nranks, trace_names=policies, log_files=log_files)
    parsed_suite = parse_suite(suite)

    prof_phases = ["app", "comp", "comm", "sync", "lb"]
    prof_data = [classify_prof_phases(nranks, df)
                 for df in parsed_suite["section_prof"]]

    plot_prof(parsed_suite, prof_phases, prof_data)
    plot_comm(parsed_suite)


def run_parse():
    nranks = 512
    trace_name = "stochsg.53"

    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    glob_patt = trace_dir_fmt.format(trace_name) + ".*"
    traces = glob.glob(glob_patt)

    policies = [ "baseline", "ci", "cdpp", "lpt", "hybrid", "hybrid02" ]
    traces = sorted(traces, key=lambda x: policies.index(x.split(".")[-1]))

    policies_hum_map = {
            "baseline": "Baseline",
            "lpt": "LPT",
            "ci": "CDP",
            "cdpp": "CDPI",
            "hybrid": "Hybrid",
            "hybrid02": "Hybrid 02"
            }

    policies_hum = [policies_hum_map[p] for p in policies]
    

    log_files = [t + "/run/log.txt" for t in traces]
    suite = RunSuite(nranks=nranks, trace_names=policies_hum, log_files=log_files)
    parsed_suite = parse_suite(suite)

    prof_phases = ["app", "comp", "comm", "sync", "lb"]
    prof_data = [classify_prof_phases(nranks, df)
                 for df in parsed_suite["section_prof"]]

    print(prof_data)
    plot_prof(parsed_suite, prof_phases, prof_data)
    plot_comm(parsed_suite)

def run_for_mochi():
    nranks = 512
    trace_name = "stochsg.53"

    trace_dir_fmt = "/mnt/ltio/parthenon-topo/stochsg.53."

    policies = [ "baseline", "lpt", "cdpp", "hybrid", "hybrid02" ]

    policies_hum_map = {
            "baseline": "Baseline",
            "lpt": "LPT",
            "ci": "CDP",
            "cdpp": "CDPI",
            "hybrid": "Hybrid (10%)",
            "hybrid02": "Hybrid (20%)"
            }

    policies_hum = [policies_hum_map[p] for p in policies]
    log_files = [ f"{trace_dir_fmt}{p}/run/log.txt" for p in policies]
    suite = RunSuite(nranks=nranks, trace_names=policies_hum, log_files=log_files)
    parsed_suite = parse_suite(suite)
    plot_comm_mochi(parsed_suite)
    pass


def run():
    plot_init_big()
    # run_parse()
    run_for_mochi()


if __name__ == '__main__':
    run()
