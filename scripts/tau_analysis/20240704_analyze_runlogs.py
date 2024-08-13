import re

import importlib

parse_preload = importlib.import_module("20240604_parse_preload")
RunSuite = parse_preload.RunSuite
get_runs = parse_preload.get_runs

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from typing import TypedDict
from common import PlotSaver


class ParsedLog(TypedDict):
    policy_name: str
    time_regts: list[float]
    time_lbts: list[float]
    time_allts: list[float]


def read_log(log_path: str) -> ParsedLog:
    policy_name = log_path.split("/")[-3].split(".")[-1]

    parsed_log: ParsedLog = {
        "policy_name": policy_name,
        "time_regts": [],
        "time_lbts": [],
        "time_allts": [],
    }

    with open(log_path, "r") as f:
        lines = f.readlines()

    lines = [l for l in lines if l.startswith("cycle=") or "LB" in l]

    is_lb_ts = False

    for l in lines:
        if "[LB]" in l:
            is_lb_ts = True
            continue
        mobj = re.search(r" wsec_step=([^ ]+)", l)
        assert mobj is not None
        ts_time = float(mobj.group(1))

        if is_lb_ts:
            parsed_log["time_lbts"].append(ts_time)
        else:
            parsed_log["time_regts"].append(ts_time)

        parsed_log["time_allts"].append(ts_time)

        is_lb_ts = False

    return parsed_log


def read_log_diff_from_prev(policy_name: str, log_path: str) -> ParsedLog:
    # policy_name = log_path.split("/")[-3].split(".")[-1]

    parsed_log: ParsedLog = {
        "policy_name": policy_name,
        "time_regts": [],
        "time_lbts": [],
        "time_allts": [],
    }

    with open(log_path, "r") as f:
        lines = f.readlines()

    lines = [l for l in lines if l.startswith("cycle=") or "LB" in l]

    is_lb_ts = False
    time_prev = 0

    for l in lines:
        if "[LB]" in l:
            is_lb_ts = True
            continue
        mobj = re.search(r" wsec_total=([^ ]+)", l)
        assert mobj is not None

        ts_time_cumul = float(mobj.group(1))
        ts_time = ts_time_cumul - time_prev
        time_prev = ts_time_cumul

        if is_lb_ts:
            parsed_log["time_lbts"].append(ts_time)
        else:
            parsed_log["time_regts"].append(ts_time)

        parsed_log["time_allts"].append(ts_time)

        is_lb_ts = False

    return parsed_log


def run_fetch_data(trace_prefix: str, policies: list[str]) -> list[ParsedLog]:
    trace_dir_fmt = "/mnt/ltio/parthenon-topo"

    all_parsed_logs: list[ParsedLog] = []

    for policy in policies:
        trace_dir = f"{trace_dir_fmt}/{trace_prefix}.{policy}"
        log_path = f"{trace_dir}/run/log.txt"

        # parsed_log = read_log(log_path)
        parsed_log = read_log_diff_from_prev(policy, log_path)

        all_parsed_logs.append(parsed_log)

    return all_parsed_logs


def plot_parsed_log_totals(parsed_logs: list[ParsedLog], plot_fname: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    labels = [pl["policy_name"] for pl in parsed_logs]

    time_total_regts = [sum(pl["time_regts"]) for pl in parsed_logs]
    time_total_lbts = [sum(pl["time_lbts"]) for pl in parsed_logs]

    bar_width = 0.35
    x = np.arange(len(labels))
    ax.bar(x, time_total_regts, bar_width, label="Regular TS", zorder=2)
    ax.bar(x + bar_width, time_total_lbts, bar_width, label="LB TS", zorder=2)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Total Time (s)")
    ax.set_title("Total Time Spent on Regular TS and LB TS")

    # grid y-minor: #ddd, both major: #bbb
    ax.grid(which="major", axis="both", color="#bbb")
    ax.grid(which="minor", axis="y", color="#ddd")

    # fmt: .1f s
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    fig.tight_layout()

    PlotSaver.save(fig, "", None, plot_fname)


def plot_parsed_logs(parsed_logs: list[ParsedLog], plot_fname: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    labels = [pl["policy_name"] for pl in parsed_logs]
    times = [pl["time_allts"] for pl in parsed_logs]

    for label, ts_times in zip(labels, times):
        ax.plot(np.cumsum(ts_times), label=label, zorder=2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time Spent on Each Timestep")

    ax.legend()

    # grid y-minor: #ddd, both major: #bbb
    ax.grid(which="major", axis="both", color="#bbb")
    ax.grid(which="minor", axis="y", color="#ddd")

    # fmt: .1f s
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f} s"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    fig.tight_layout()

    PlotSaver.save(fig, "", None, plot_fname)


def run():
    trace_dir_fmt = "/mnt/ltio/parthenon-topo"
    trace_name = "blastw4096.01"

    policies = ["baseline", "cdp", "hybrid25", "hybrid50", "hybrid75", "lpt"]
    parsed_logs = run_fetch_data(trace_name, policies)
    plot_parsed_log_totals(parsed_logs, f"ts_times_from_prev_{trace_name}")


def run_suite():
    nranks = 4096
    run_id = f"blastw{nranks}.04"
    fallback_run_id = f"blastw{nranks}.03"

    suite = get_runs(nranks, run_id, fallback_run_id)
    parsed_logs: list[ParsedLog] = [
        read_log_diff_from_prev(r, l)
        for r, l in zip(suite.trace_names, suite.log_files)
    ]
    run_id = suite.run_id
    plot_parsed_log_totals(parsed_logs, f"ts_times_from_prev_{suite.run_id}")
    plot_parsed_logs(parsed_logs, f"ts_times_line_{suite.run_id}")


if __name__ == "__main__":
    # run()
    run_suite()
