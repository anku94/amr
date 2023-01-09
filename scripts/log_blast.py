#!/usr/bin/env python

import h5py
import numpy as np
import sys

from glob import glob
from typing import List

def get_data(i: int):
    with h5py.File(filenames[i], "r") as f:
        xf = f["Locations"]["x"][0]
        xc = 0.5 * (xf[1:] + xf[:-1])
        P = f["pressure"][0, 0, 0, :, 0]
        tau = f["p.energy"][0, 0, 0, :, 0]
        rho = f["p.density"][0, 0, 0, :, 0]
        D = f["c.density"][0, 0, 0, :, 0]
        v = f["p.velocity"][0, 0, 0, :, 0]
    return xc, P, tau, rho, D, v


def log_all_data(filenames: List[str]):
    np.set_printoptions(precision=2, linewidth=120)

    for i in range(len(filenames)):
        xc, P, tau, rho, D, v = get_data(i)
        print(f"[{i}] XC: {xc}")
        print(f"[{i}] rho: {rho}")


def run(filenames: List[str]):
    log_all_data(filenames)


if __name__ == "__main__":
    data_root = sys.argv[1]
    filenames = sorted(glob(f'{data_root}/sedov.out1.*.phdf'))
    print(f'[Data Root] Scanning {data_root}...')
    print(f'{len(filenames)} files found!')
    run(filenames)
