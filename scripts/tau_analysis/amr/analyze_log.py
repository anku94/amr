import re

from typing import TypedDict
from .trace_common import TraceUtils

import numpy as np
import pandas as pd


class LogLine(TypedDict):
    is_lb: bool
    cycle: int
    time: float
    dt: float
    wsec_total: float
    wsec_step: float
    wsec_amr: float


class AmrmonData(TypedDict):
    keys: list[str]
    data: list[np.ndarray]


"""
parse_line: does not set is_lb, user should set it
"""


def parse_line(line: str) -> LogLine:
    parts = line.split(" ")
    parsed_times: dict[str, float] = {}

    for p in parts:
        pk, pv = p.split("=")
        parsed_times[pk] = float(pv)

    log_line: LogLine = {
        "is_lb": False,
        "cycle": int(parsed_times["cycle"]),
        "time": parsed_times["time"],
        "dt": parsed_times["dt"],
        "wsec_total": parsed_times["wsec_total"],
        "wsec_step": parsed_times["wsec_step"],
        "wsec_amr": parsed_times["wsec_AMR"],
    }

    return log_line


def read_log(log_path: str) -> pd.DataFrame:
    with open(log_path, "r") as f:
        lines = f.readlines()

    ts_re = re.compile(r"^cycle.*$")
    lb_re = re.compile(r"\[LB\] [^  ]* being invoked!$")

    lines = [l for l in lines if ts_re.match(l) or lb_re.match(l)]

    matched_lines: list[LogLine] = []

    is_lb = False
    for l in lines:
        if lb_re.match(l):
            is_lb = True
            continue

        if ts_re.match(l):
            parsed_line = parse_line(l)
            parsed_line["is_lb"] = is_lb
            matched_lines.append(parsed_line)
            is_lb = False

    return pd.DataFrame(matched_lines)


def read_amrmon_single(fpath: str) -> AmrmonData:
    raw_data = open(fpath, "r").readlines()
    raw_data = [x.strip() for x in raw_data]

    phases = raw_data[::2]
    phase_data = raw_data[1::2]
    str2arr = lambda x: np.array([float(x) for x in x.split(",")])

    amrmon_data: AmrmonData = {
        "keys": phases,
        "data": [str2arr(v) for v in phase_data],
    }

    return amrmon_data


def read_suite_logs(trace_key: str) -> list[pd.DataFrame]:
    suite = TraceUtils.get_traces(trace_key)
    log_files = suite.log_files()
    return [read_log(log_path) for log_path in log_files]


def read_suite_amrmon(trace_key: str) -> list[AmrmonData]:
    suite = TraceUtils.get_traces(trace_key)
    amrmon_files = suite.amrmon_logs()
    amrmon_data: list[AmrmonData] = [
        read_amrmon_single(amrmon_path) for amrmon_path in amrmon_files
    ]
    return amrmon_data


def run():
    trace_key = "mpidbg4096ub22.tune1"
    suite = TraceUtils.get_traces(trace_key)
    log_files = suite.log_files()
    print(suite)
    print(log_files)
    log_path = log_files[0]
    lines = read_log(log_path)
    df = pd.DataFrame(lines)
    df
    lines
    lines[0]
    line = lines[0][1]
    pass


if __name__ == "__main__":
    run()
