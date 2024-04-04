import numpy as np
import re
from common import PlotSaver

import matplotlib.pyplot as plt

def get_times(run_dir):
    fpath = f"{run_dir}/run/log.txt"
    data = open(fpath, 'r').read().split('\n')
    lines = [l.strip() for l in data if l.startswith('cycle=')]

    cycles = []
    times = []

    for l in lines:
        mobj = re.match(r'cycle=(\d+).*wsec_step=([^\ ]+).*$', l)
        c = int(mobj.group(1))
        t = float(mobj.group(2))
        
        cycles.append(c)
        times.append(t)

    return (cycles, times)

def plot_times(times):
    data_x, data_y = times
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ylim = np.percentile(data_y, 97)

    ax.plot(data_x, data_y)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.set_title("StochSubgrid - Time per Timestep")
    ax.grid()

    ax.set_ylim([0, ylim])

    fig.tight_layout()
    plot_fname = "stochsg_times"
    PlotSaver.save(fig, "", None, plot_fname)

def run():
    run_dir = "/mnt/ltio/parthenon-topo/stochsg5"
    times = get_times(run_dir)
    times
    plot_times(times)
    pass

if __ name__ == "__main__":
    run()
