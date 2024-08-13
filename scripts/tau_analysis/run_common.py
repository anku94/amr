from trace_common import all_trace_sets, TraceSet

import glob
import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def dump_traceset(runs: dict[str, TraceSet], fout: str):
    with open(fout, "w") as f:
        yaml.dump(runs, f, Dumper=NoAliasDumper, sort_keys=True)
        print(f"Wrote all runs to {fout}")


def read_traceset(fin: str) -> dict[str, TraceSet]:
    typed_runs: dict[str, TraceSet] = {}

    with open(fin, "r") as f:
        runs = yaml.safe_load(f)
        print(f"Read all runs from {fin}")

    for k, v in runs.items():
        typed_runs[k] = TraceSet(**v)

    return typed_runs


def read_all_tracesets() -> dict[str, TraceSet]:
    yaml_root = "/users/ankushj/repos/amr-new/scripts/tau_analysis/all_tracesets"
    yaml_files = glob.glob(f"{yaml_root}/*.yaml")
    all_dicts: list[dict[str, TraceSet]] = [read_traceset(f) for f in yaml_files]

    merged_dict: dict[str, TraceSet] = {k: v for d in all_dicts for k, v in d.items()}
    return merged_dict
