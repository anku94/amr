#!/usr/bin/env python3

import argparse
import os

from pathlib import Path
from typing import Any, Dict, Generator, List, TypedDict, Optional


class TopobenchTemplateOpts(TypedDict):
    job_name: str
    ranks: int
    nodes: int
    ppn: int
    meshgen_method: int
    blocks_per_rank: int
    msgsz: int
    num_timesteps: int
    num_rounds: int
    topo_trace_name: str
    cpubind: str
    topo_trace_path: str
    hostsuffix: str
    iterations: int
    script: str


class TopobenchSuite(TypedDict):
    all_nranks: List[int]
    all_nrounds: List[int]
    all_traces: List[str]
    trace_fname_fmt: str
    meshgen_method: int
    blocks_per_rank: int
    msgsz: int


basic_suite: TopobenchSuite = {
    "all_nranks": [512, 1024, 2048, 4096],
    "all_nrounds": [1, 2, 3, 4, 5],
    "all_traces": ["baseline.uniform", "baseline.reg", "lpt.uniform", "lpt.reg"],
    "trace_fname_fmt": "traces/topobench/singlets/blastw.{nranks}.{trace_name}.csv",
    "meshgen_method": 4,
    "blocks_per_rank": -1,
    "msgsz": -1,
}

mini_suite: TopobenchSuite = {
    "all_nranks": [512],
    "all_nrounds": [1, 2, 3],
    "all_traces": ["baseline.uniform"],
    "trace_fname_fmt": "traces/topobench/singlets/blastw.{nranks}.{trace_name}.csv",
    "meshgen_method": 4,
    "blocks_per_rank": -1,
    "msgsz": -1,
}

all_suites: Dict[str, TopobenchSuite] = {
    "basic": basic_suite,
    "mini": mini_suite,
}


class CLIOptions(TypedDict):
    ppn: int
    topo_trace_path: str
    cpubind: str
    hostsuffix: str
    iterations: int
    script: str


class TopobenchCLIOption(TypedDict):
    name: str
    default: Any
    help: str
    choices: Optional[List]


default_opts: List[TopobenchCLIOption] = [
    {"name": "ppn", "default": 16, "help": "Procs per node", "choices": None},
    {
        "name": "topo_trace_path",
        "default": "traces/topobench/singlets",
        "help": "Prefix for trace files",
        "choices": None,
    },
    {
        "name": "cpubind",
        "default": "none",
        "help": "CPU binding",
        "choices": ["none", "core", "thread", "external"],
    },
    {
        "name": "hostsuffix",
        "default": "",
        "help": "Suffix for hostnames",
        "choices": None,
    },
    {
        "name": "iterations",
        "default": 1,
        "help": "Number of iterations",
        "choices": None,
    },
    {
        "name": "script",
        "default": "run_topo_test.sh",
        "help": "Name of the script",
        "choices": None,
    },
]


def setup_opts(parser: argparse.ArgumentParser):
    for opt in default_opts:
        opt_name_arg = opt["name"].replace("_", "-")
        parser.add_argument(
            f"--{opt_name_arg}",
            type=str,
            default=opt["default"],
            help=opt["help"],
            choices=opt["choices"],
        )


def get_opts(args: argparse.Namespace) -> CLIOptions:
    return {
        "ppn": args.ppn,
        "topo_trace_path": args.topo_trace_path,
        "cpubind": args.cpubind,
        "hostsuffix": args.hostsuffix,
        "iterations": args.iterations,
        "script": args.script,
    }


def suite_to_jobs(
    suite: TopobenchSuite,
    common_opts: CLIOptions,
) -> Generator[TopobenchTemplateOpts, None, None]:
    job_name_fmt = "topobench_{nranks}_{nrounds}_{trace_name}"
    trace_name_fmt = suite["trace_fname_fmt"]

    for nranks in suite["all_nranks"]:
        for nrounds in suite["all_nrounds"]:
            for trace_name in suite["all_traces"]:
                yield {
                    "job_name": job_name_fmt.format(
                        nranks=nranks, nrounds=nrounds, trace_name=trace_name
                    ),
                    "ranks": nranks,
                    "nodes": nranks // common_opts["ppn"],
                    "ppn": common_opts["ppn"],
                    "meshgen_method": suite["meshgen_method"],
                    "blocks_per_rank": suite["blocks_per_rank"],
                    "msgsz": suite["msgsz"],
                    "num_timesteps": 1,
                    "num_rounds": nrounds,
                    "topo_trace_name": trace_name_fmt.format(
                        nranks=nranks, trace_name=trace_name
                    ),
                    "cpubind": common_opts["cpubind"],
                    "topo_trace_path": common_opts["topo_trace_path"],
                    "hostsuffix": common_opts["hostsuffix"],
                    "iterations": common_opts["iterations"],
                    "script": common_opts["script"],
                }


def get_template(template_file: str) -> str:
    with open(template_file, "r") as f:
        return f.read()


def write_job(template_str: str, job: TopobenchTemplateOpts, job_fpath: Path) -> None:
    with open(job_fpath, "w") as f:
        f.write(template_str.format_map(job))

    # Make the file executable
    os.chmod(job_fpath, 0o755)

    return


def run_gen(
    template_file: str,
    suite: TopobenchSuite,
    common_opts: CLIOptions,
    output_dir: Path,
) -> None:
    template_str = get_template(template_file)

    os.makedirs(output_dir, exist_ok=True)

    for job in suite_to_jobs(suite, common_opts):
        job_name = job["job_name"]
        job_fpath = output_dir / f"{job_name}.sh"
        print(f"-INFO- Generating job for {job_name} at {job_fpath}")

        write_job(template_str, job, job_fpath)

    return


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template-file",
        type=Path,
        required=True,
        help="Template file for job script",
    )

    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Suite to run",
        choices=all_suites.keys(),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write job files to",
    )

    setup_opts(parser)
    args = parser.parse_args()

    template_file = args.template_file
    output_dir = args.output_dir
    common_opts = get_opts(args)
    suite = mini_suite

    if not template_file.exists():
        raise FileNotFoundError(f"Template file {template_file} does not exist")

    if not output_dir.parent.exists():
        raise FileNotFoundError(f"Parent directory {output_dir.parent} does not exist")

    if not output_dir.exists():
        os.makedirs(output_dir)

    run_gen(template_file, suite, common_opts, output_dir)


if __name__ == "__main__":
    run()
