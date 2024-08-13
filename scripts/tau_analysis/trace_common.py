import glob
import os
import re

from dataclasses import dataclass
from typing import TypedDict

RunTuple = tuple[str, str | None, str]


class TraceSet(TypedDict):
    prefix: str
    nranks: int
    run_ids: dict[str, str]
    run_tuples: list[RunTuple]
    desc: str


@dataclass
class SingleTrace:
    name: str
    lb_policy: str
    path: str

    def get_log(self) -> str:
        return f"{self.path}/run/log.txt"

    def get_amrmon_log(self) -> str:
        return f"{self.path}/trace/amrmon_rankwise.txt"

    def get_tau_dir(self) -> str:
        glob_patt = f"{self.path}/**/profile.*"
        profiles = glob.glob(glob_patt, recursive=True)
        profile_patt = lambda x: re.match(r"profile.[0-9]+\.0.0", os.path.basename(x))
        profiles = list(filter(profile_patt, profiles))
        profile_parents = set([os.path.dirname(x) for x in profiles])
        assert len(profile_parents) == 1
        return list(profile_parents)[0]


@dataclass
class TraceSuite:
    suite_id: str
    nranks: int
    traces: list[SingleTrace]

    def __str__(self):
        suite_repr = f"TraceSuite(nranks={self.nranks})"
        for trace in self.traces:
            suite_repr += f"\n -{trace.lb_policy}: {trace.path}"

        return suite_repr

    def trace_names(self) -> list[str]:
        return [trace.name for trace in self.traces]

    def log_files(self) -> list[str]:
        return [trace.get_log() for trace in self.traces]

    def amrmon_logs(self) -> list[str]:
        return [trace.get_amrmon_log() for trace in self.traces]


all_trace_sets: dict[str, TraceSet] = {
    "cdppar4096.first": {
        "desc": """Evaluating cdpc512par8 on 20240715 with 4096 ranks""",
        "prefix": "/mnt/ltio/parthenon-topo",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.03", "b": "blastw4096.04"},
        "run_tuples": [
            ("baseline", None, "a"),
            # ("cdp", None, "a"),
            # ("cdpc512", None, "b"),
            ("cdpc512par8", None, "b"),
            # ("hybrid25_old", "hybrid25", "a"),
            ("hybrid25", None, "b"),
            # ("hybrid50_old", "hybrid50", "a"),
            ("hybrid50", None, "b"),
            # ("hybrid75_old", "hybrid75", "a"),
            ("hybrid75", None, "b"),
            ("lpt", None, "a"),
        ],
    },
    "cdppar2048": {
        "desc": """Evaluating cdpc512par8 on 20240715 with 2048 ranks,
        as previous run did not make much sense
        """,
        "prefix": "/mnt/ltio/parthenon-topo/mpiallreddbg",
        "nranks": 2048,
        "run_ids": {"a": "blastw2048.02.preload"},
        "run_tuples": [
            ("baseline", None, "a"),
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
            ("hybrid50", None, "a"),
            ("hybrid75", None, "a"),
            ("lpt", None, "a"),
        ],
    },
    "mpidbgwbar": {
        "desc": "4096 ranks, 10k ts, added MPI_Barrier at the end of R&RMB",
        "prefix": "/mnt/ltio/parthenon-topo/mpiallreddbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.03.preload"},
        "run_tuples": [("cdpc512par8", None, "a"), ("hybrid25", None, "a")],
    },
    "mpidbg2048ub22": {
        "desc": "2048 ranks, 10k ts, barriers, on Ubuntu 22.04, with tauexec",
        "prefix": "/proj/TableFS/ankushj/amr-jobs",
        "nranks": 2048,
        "run_ids": {"a": "blastw2048.perfdbg.01"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
            ("hybrid75", None, "a"),
        ],
    },
    "mpidbg4096on2048ub22": {
        "desc": "2048 ranks, 4096 deck, 10k ts, barriers, on Ubuntu 22.04, with tauexec",
        "prefix": "/proj/TableFS/ankushj/amr-jobs",
        "nranks": 2048,
        "run_ids": {"a": "blastw2048.02.4096deck"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
            ("hybrid50", None, "a"),
            ("hybrid75", None, "a"),
            ("lpt", None, "a"),
        ],
    },
    "mpidbg4096ub22": {
        "desc": "4096 ranks, 10k ts, barriers, on Ubuntu 22.04, with tauexec",
        "prefix": "/proj/TableFS/ankushj/amr-jobs",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.02.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid50", None, "a"),
            ("lpt", None, "a"),
        ],
    },
    "mpidbg4096ub22.03": {
        "desc": "4096 ranks, 1k ts, barriers, on Ubuntu 22.04, with tauexec",
        "prefix": "/proj/TableFS/ankushj/amr-jobs",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.03.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
            ("hybrid50", None, "a"),
            ("hybrid75", None, "a"),
            ("lpt", None, "a"),
        ],
    },
    "mpidbg4096ub22.04": {
        "desc": "4096 ranks, 10k ts, barriers, on Ubuntu 22.04, with tauexec",
        "prefix": "/proj/TableFS/ankushj/amr-jobs",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.04.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
            ("hybrid50", None, "a"),
            ("hybrid75", None, "a"),
            ("lpt", None, "a"),
        ],
    },
    "mpidbg4096ub22.05": {
        "desc": "4096 ranks, 2k ts, barriers, on Ubuntu 22.04, with tauexec",
        "prefix": "/proj/TableFS/ankushj/amr-jobs",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.05.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
        ],
    },
    "mpidbg4096ub22.06": {
        "desc": "4096 ranks, 2k ts, barriers, on Ubuntu 22.04, with perf+libperf",
        "prefix": "/mnt/ltio/parthenon-topo/ub22perfdbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.06.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
        ],
    },
    "mpidbg4096ub22.07": {
        "desc": "4096 ranks, 2k 1s, barriers, on Ubuntu 22.04, with perf+libperf",
        "prefix": "/mnt/ltio/parthenon-topo/ub22perfdbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.07.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
        ],
    },
    "mpidbg4096ub22.08": {
        "desc": "4096 ranks, 2k 1s, barriers, on Ubuntu 22.04, with perf+libperf",
        "prefix": "/mnt/ltio/parthenon-topo/ub22perfdbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.08.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid25", None, "a"),
        ],
    },
    "mpidbg4096ub22.12": {
        "desc": "4096 ranks, all ts, barriers, on Ubuntu 22.04, libperf",
        "prefix": "/mnt/ltio/parthenon-topo/ub22perfdbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.12.perfdbg"},
        "run_tuples": [
            ("baseline", None, "a"),
            ("cdpc512par8", None, "a"),
            ("hybrid10", None, "a"),
            ("hybrid25", None, "a"),
            ("lpt", None, "a"),
        ],
    },
    "mpidbg4096ub22.13": {
        "desc": "4096 ranks, ?? ts, h25 vs LPT amrmon-tswise",
        "prefix": "/mnt/ltio/parthenon-topo/ub22perfdbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.13.perfdbg"},
        "run_tuples": [
            ("hybrid25", None, "a"),
        ],
    },
    "mpidbg4096ub22.18": {
        "desc": "4096 ranks, 10k ts, with caching, h25 vs LPT amrmon-tswise",
        "prefix": "/mnt/ltio/parthenon-topo/ub22perfdbg",
        "nranks": 4096,
        "run_ids": {"a": "blastw4096.18.perfdbg"},
        "run_tuples": [
            ("cdpc512par8", None, "a"),
            ("hybrid10", None, "a"),
            ("hybrid25", None, "a"),
            ("lpt", None, "a"),
        ],
    },
}

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


class TraceUtils:
    @staticmethod
    def desc_traces():
        for k in all_trace_sets:
            print(f"{k}: {all_trace_sets[k]['desc']}")

    @staticmethod
    def get_traces(trace_key: str) -> TraceSuite:
        traces: list[SingleTrace] = []
        trace_set = all_trace_sets[trace_key]

        run_paths: dict[str, str] = {}

        for run_alias, run_id in trace_set["run_ids"].items():
            run_path = f"{trace_set['prefix']}/{run_id}"
            run_paths[run_alias] = run_path

        for name, policy, run_alias in trace_set["run_tuples"]:
            if policy is None:
                policy = name

            path = f"{run_paths[run_alias]}.{policy}"

            traces.append(SingleTrace(name, policy, path))

        trace_suite: TraceSuite = TraceSuite(trace_key, trace_set["nranks"], traces)

        return trace_suite


if __name__ == "__main__":
    traces = TraceUtils.get_traces("mpidbg2048ub22")
    t = traces.traces[0]
    p = t.get_tau_dir()
    print(t)
    print(p)
