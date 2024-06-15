#!/usr/bin/env python3

import argparse
import os

import pandas as pd

from pathlib import Path
from typing import Generator, List, Literal, TypedDict

LoadBalancingPolicy = Literal[
    "baseline", "cdp", "hybrid25", "hybrid50", "hybrid75", "lpt"
]


class AMRTemplateOpts(TypedDict):
    job_name: str
    amr_bin: str
    amr_deck: str
    ranks: int
    procs_per_node: int
    nodes: int
    lb_policy: LoadBalancingPolicy
    nlim: int
    preload: str
    niters: int
    job_time: str


FNAME_SUFFIX = ".in"  # .in in build tree, nothing in installt ree


"""
Compute this script's file path
"""


def compute_own_path() -> Path:
    return Path(__file__).resolve().parent


def compute_suffix_using_heuristics():
    print("-INFO- Using heuristics to detect if in build tree or install tree")

    global FNAME_SUFFIX

    own_path = compute_own_path()
    # get all files in this directory
    files = os.listdir(own_path)
    if "run_amr_test.sh.in" in files:
        print("-INFO- Detected build tree. " "Generated files will have .in suffix")
        FNAME_SUFFIX = ".in"
    elif "run_amr_test.sh" in files:
        print(
            "-INFO- Detected install tree." "Generated files will not have .in suffix"
        )
        FNAME_SUFFIX = ""
    else:
        raise FileNotFoundError("Could not detect build or install tree")

    return


def get_template(template_fname: str) -> str:
    with open(template_fname, "r") as f:
        template_str = f.read()

    return template_str


def write_job(template: str, vars: AMRTemplateOpts, fpath_out: str) -> None:
    with open(fpath_out, "w+") as f:
        job_str = template.format_map(vars)
        f.write(job_str)

    # Make the file executable
    os.chmod(fpath_out, 0o755)

    return


def row_to_jobs(row) -> Generator[AMRTemplateOpts, None, None]:
    global FNAME_SUFFIX

    job_obj: AMRTemplateOpts = {
        "job_name": row["job_name"],
        "amr_bin": row["amr_bin"],
        "amr_deck": row["amr_deck"],
        "ranks": row["ranks"],
        "procs_per_node": row["procs_per_node"],
        "nodes": row["nodes"],
        "lb_policy": row["lb_policy"],
        "nlim": row["nlim"],
        "preload": row["preload"],
        "niters": row["niters"],
        "job_time": row["job_time"],
    }

    nranks = job_obj["ranks"]
    nnodes = job_obj["nodes"]
    npn = job_obj["procs_per_node"]
    lb_policy = job_obj["lb_policy"]

    assert nranks == nnodes * npn

    abs_path = compute_own_path()
    deck_dir_cand1 = (abs_path / ".." / "decks").resolve()
    deck_dir_cand2 = (abs_path / ".." / ".." / "decks").resolve()

    if deck_dir_cand1.exists():
        deck_dir = deck_dir_cand1
    elif deck_dir_cand2.exists():
        deck_dir = deck_dir_cand2
    else:
        raise FileNotFoundError("Could not find decks directory")

    amr_deck = job_obj["amr_deck"]
    # Deck name will have scale as a matter of convention
    amr_deck_scale = int(amr_deck.split(".")[-2])
    if amr_deck_scale != job_obj["ranks"]:
        raise ValueError("Deck scale does not match number of ranks")

    deck_fpath = deck_dir / f"{amr_deck}.in"
    print(f"-INFO- Checking if deck {deck_fpath.name} exists ...")
    assert deck_fpath.exists()

    if job_obj["lb_policy"] != "<all>":
        yield job_obj

    all_lb_policies: List[LoadBalancingPolicy] = [
        "baseline",
        "cdp",
        "hybrid25",
        "hybrid50",
        "hybrid75",
        "lpt",
    ]

    for lb_policy in all_lb_policies:
        job_obj["lb_policy"] = lb_policy
        job_obj["job_name"] = f"{row['job_name']}-{nranks}-{lb_policy}"
        yield job_obj

    return


def run_gen(template_file: str, input_csv: str, output_dir: str) -> None:
    global FNAME_SUFFIX

    template_str = get_template(template_file)

    input_csv = "amr_suite.csv"
    df = pd.read_csv(input_csv)

    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        row_jobs = row_to_jobs(row)
        for job in row_jobs:
            job_name = job["job_name"]
            job_fpath = f"{output_dir}/{job_name}.sh{FNAME_SUFFIX}"
            print(f"-INFO- Generating job for {job_name} at {job_fpath}")

            write_job(template_str, job, job_fpath)

    return


def run():
    parser = argparse.ArgumentParser(description="Generate AMR job scripts")

    parser.add_argument(
        "-f",
        "--file-template",
        type=str,
        help="Template file for job script",
        required=True,
    )

    parser.add_argument(
        "-i",
        "--input-csv",
        type=str,
        help="CSV file containing AMR job parameters",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for generated job scripts",
        required=True,
    )

    args = parser.parse_args()

    print(f"-INFO- Generating AMR job scripts at {args.output_dir}")

    compute_suffix_using_heuristics()
    run_gen(args.file_template, args.input_csv, args.output_dir)


#

if __name__ == "__main__":
    run()
