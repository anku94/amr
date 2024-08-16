import glob
import os
import re

from dataclasses import dataclass

from .run_common import TraceSet, read_all_tracesets


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


all_trace_sets: dict[str, TraceSet] = read_all_tracesets()

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

            path = f"{run_paths[run_alias]}.{name}"

            traces.append(SingleTrace(name, policy, path))

        trace_suite: TraceSuite = TraceSuite(trace_key, trace_set["nranks"], traces)

        return trace_suite


if __name__ == "__main__":
    traces = TraceUtils.get_traces("mpidbg2048ub22")
    t = traces.traces[0]
    p = t.get_tau_dir()
    print(t)
    print(p)
