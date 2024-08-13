import multiprocessing
import subprocess

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from common import PlotSaver, plot_init_big as plot_init
from dataclasses import dataclass
from io import StringIO


@dataclass
class PerfChunk:
    fpath: str
    ranks: list[int]

    def __str__(self):
        return f"PerfChunk({min(self.ranks)}, {max(self.ranks)})"


def read_fpath(fpath: str) -> str:
    cmd_fmt = "perf report --no-inline -i {fpath} --stdio --max-stack 1 | grep phoebus"

    cmd = cmd_fmt.format(fpath=fpath)
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    data = stdout.decode("utf-8")

    return data


def parse_data(data: str) -> pd.DataFrame:
    pbeg = data.index("phoebus")
    pend = pbeg + 7
    kbeg = data.index("[.]")
    kend = kbeg + 4

    colspecs = [
        (0, pbeg - 1),
        (pbeg, pend),
        (pend + 1, kbeg - 1),
        (kbeg, kend),
        (kend, None),
    ]
    cols = ["pct", "exe", "so", "k", "func"]
    df = pd.read_fwf(StringIO(data), colspecs=colspecs, names=cols)

    df_k = df["k"].unique().tolist()
    assert len(df_k) == 2 and "[k]" in df_k and "[.]" in df_k

    df_exe = df["exe"].unique().tolist()
    assert len(df_exe) == 1 and "phoebus" in df_exe

    df.drop(columns=["k", "exe"], inplace=True)
    df.pct = df.pct.str.rstrip("%").astype(float)

    df.func = df.func.str.strip(" \n")

    return df


def run_chunk(perf_chunk: PerfChunk) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []

    for rank in perf_chunk.ranks:
        if rank % 2 == 0:
            print(f"Processing rank {rank}")

        fpath = f"{perf_chunk.fpath}/prof/perf.data.{rank}"
        data = read_fpath(fpath)
        df = parse_data(data)
        df["rank"] = rank
        df = df[["rank", "so", "func", "pct"]]
        assert type(df) == pd.DataFrame

        all_dfs.append(df)

    df_concat = pd.concat(all_dfs)
    return df_concat


def gen_chunks(fpath: str, nranks: int, nchunks: int) -> list[PerfChunk]:
    chunk_sz = nranks // nchunks
    cbegs = [i * chunk_sz for i in range(nchunks)]
    cends = [i * chunk_sz for i in range(1, nchunks)] + [nranks]
    pairs = list(zip(cbegs, cends))

    all_chunks: list[PerfChunk] = []
    for cbeg, cend in pairs:
        ranks = list(range(cbeg, cend))
        chunk = PerfChunk(fpath=fpath, ranks=ranks)
        all_chunks.append(chunk)

    return all_chunks


def run_parse_parallel(perf_dir: str, nranks: int, nchunks: int) -> pd.DataFrame:
    all_chunks = gen_chunks(perf_dir, nranks, nchunks)

    with multiprocessing.Pool(nchunks) as pool:
        all_data = pool.map(run_chunk, all_chunks)

    df_concat = pd.concat(all_data)
    return df_concat


def parse_and_gen(perf_dir: str, nranks: int, nchunks: int):
    df = run_parse_parallel(perf_dir, nranks, nchunks)
    df.to_feather(f"{perf_dir}/perf.feather")


def read_perf_data(perf_dir: str) -> pd.DataFrame:
    df = pd.read_feather(f"{perf_dir}/perf.feather")
    return df


def plot_ax_prof_aggr(ax: plt.Axes, df: pd.DataFrame):
    data_mean = df.head(10)["pct_mean"]
    data_min = df.head(10)["pct_min"]
    data_max = df.head(10)["pct_max"]
    data_name = df.head(10)["func"]

    def parse_label(label: str) -> str:
        if "<" in label:
            return label.split("<")[0]

        return label

    xlabels = data_name.map(parse_label)

    ax.clear()

    bar_width = 0.25
    data_x = np.arange(len(data_mean))
    bar_ret = ax.bar(data_x, data_mean, bar_width, label="mean", zorder=2)
    ax.bar_label(bar_ret, labels=[f"{x:.0f}%" for x in data_mean], padding=3)

    bar_ret = ax.bar(data_x + bar_width, data_min, bar_width, label="min", zorder=2)
    ax.bar_label(bar_ret, labels=[f"{x:.0f}%" for x in data_min], padding=3)

    bar_ret = ax.bar(data_x + 2 * bar_width, data_max, bar_width, label="max", zorder=2)
    ax.bar_label(bar_ret, labels=[f"{x:.0f}%" for x in data_max], padding=3)

    ax.set_xticks(data_x + bar_width)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.grid(which="major", color="#bbb")
    ax.yaxis.grid(which="minor", color="#ddd")

    ax.set_ylim([0, 30])


def compare_func_inner(cdp: pd.DataFrame, h25: pd.DataFrame, func: str, ax: plt.Axes):
    cdp_func = cdp[cdp["func"] == func]
    h25_func = h25[h25["func"] == func]

    cdp_vals = cdp_func.sort_values("rank")["pct"].to_numpy()
    h25_vals = h25_func.sort_values("rank")["pct"].to_numpy()

    ax.clear()
    ax.plot(cdp_vals, label="cdp", zorder=2)
    ax.plot(h25_vals, label="h25", zorder=2)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.grid(which="major", color="#bbb")
    ax.grid(which="minor", color="#ddd")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.legend()


def get_func_arr(df: pd.DataFrame, func: str, nranks: int):
    func_arr = df[df["func"] == func].sort_values("rank")[["rank", "pct"]]
    func_arr = func_arr.set_index("rank").reindex(range(nranks))
    func_arr = func_arr.fillna(0).to_numpy().flatten()
    return func_arr


def get_low_outliers(df: pd.DataFrame, recvfunc: str, nranks: int) -> list[int]:
    func_arr = get_func_arr(df, recvfunc, nranks)

    pct1 = np.percentile(func_arr, 1)
    outliers = np.where(func_arr < pct1)[0]
    outlier_data = func_arr[outliers]
    outlier_data = list(zip(outliers, outlier_data))
    outlier_data = sorted(outlier_data, key=lambda x: x[1])

    print(f"Top 10 outliers for recvfunc: ")
    for i, (rank, pct) in enumerate(outlier_data[:10]):
        print(f" - {i+1:2d}. Rank {rank:4d}: {pct:.2f}%")

    top_10 = [x[0] for x in outlier_data[:10]]
    return top_10


def analyze(perf_dir_cdp: str, perf_dir_h25: str, nranks: int):
    nranks = 4096

    cdp = read_perf_data(perf_dir_cdp)
    h25 = read_perf_data(perf_dir_h25)

    cdpg = cdp.groupby("func", as_index=False).agg({"pct": ["min", "max", "mean"]})
    cdpg.columns = ["func", "pct_min", "pct_max", "pct_mean"]
    cdpg.sort_values("pct_mean", ascending=False, inplace=True)

    h25g = h25.groupby("func", as_index=False).agg({"pct": ["min", "max", "mean"]})
    h25g.columns = ["func", "pct_min", "pct_max", "pct_mean"]
    h25g.sort_values("pct_mean", ascending=False, inplace=True)

    assert type(cdpg) == pd.DataFrame
    assert type(h25g) == pd.DataFrame

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_ax_prof_aggr(axes[0], cdpg)
    plot_ax_prof_aggr(axes[1], h25g)
    plot_fname = "perf.cdp_vs_h25.blastw4096.05"
    PlotSaver.save(fig, "", None, plot_fname)

    all_funcs = cdpg.func.iloc[:10].tolist()
    all_func_names = [
        "mv2_shm_barrier",
        "ips_ptl_shared_poll",
        "psm_mq_ipeek",
        "memset_sse",
        "con2prim::solve",
        "psm_progress_wait",
        "con2prim::residual",
        "send_bb",
        "set_bounds",
        "recv_bb",
    ]

    memset_arr = get_func_arr(cdp, all_funcs[3], nranks)
    memset_arr[3448:3456]
    3448 / 16.0

    # recv outliers
    cdp_10 = get_low_outliers(cdp, all_funcs[-1], nranks)
    h25_10 = get_low_outliers(h25, all_funcs[-1], nranks)
    cdp_10
    h25_10
    common_outliers = list(set(cdp_10) & set(h25_10))
    common_outliers

    compute_rank_ptile = lambda arr, x: np.sum(arr < arr[x]) / len(arr)

    for name, func in zip(all_func_names, all_funcs):
        print(f"Function: {name}")
        for r in common_outliers:
            cdp_arr = get_func_arr(cdp, func, nranks)
            h25_arr = get_func_arr(h25, func, nranks)

            cdp_ptile = compute_rank_ptile(cdp_arr, r)
            h25_ptile = compute_rank_ptile(h25_arr, r)

            print(f" - Rank {r:4d}: {cdp_ptile:2.2%} (cdp) vs {h25_ptile:2.2%} (h25)")
        
    for df_key in df_keys:
        df = df_map[df_key]
        for r in common_outliers:
            print(f"Rank {r} in {df_key}:")
            # for name, func in zip(all_func_names, all_funcs):
            func_arr = get_func_arr(df, func, nranks)
            rank_ptile = compute_rank_ptile(func_arr, r)
            print(f" - {name:35s}: {rank_ptile:.2%}")

        # get percentile of func_arr[ranktotarget] in func_arr

    recvfunc = all_funcs[-1]

    for name, func in zip(all_func_names, all_funcs):
        fig, ax = plt.subplots(1, 1, figsize=(32, 8))
        compare_func_inner(cdp, h25, func, ax)
        ax.set_title(f"func: {name}")
        plot_fname = f"perf.cdp_vs_h25.{name}.blastw4096.05"
        print("Saving", plot_fname)
        PlotSaver.save(fig, "", None, plot_fname)
        plt.close(fig)


def run():
    perf_run = "/mnt/ltio/parthenon-topo/ub22perfdbg"
    perf_dir_cdp = f"{perf_run}/blastw4096.05.perfdbg.cdpc512par8"
    perf_dir_h25 = f"{perf_run}/blastw4096.05.perfdbg.hybrid25"

    nranks = 4096
    nchunks = 16

    # parse_and_gen(perf_dir_cdp, nranks, nchunks)
    # parse_and_gen(perf_dir_h25, nranks, nchunks)

    analyze(perf_dir_cdp, perf_dir_h25)


if __name__ == "__main__":
    plot_init()
    run()
