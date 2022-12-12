import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pickle
import ipdb

from trace_reader import TraceReader, TraceOps


def plot_init():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_neighbors(df, plot_dir):
    fig, ax = plt.subplots(1, 1)
    print(df.describe())
    print(df.columns)

    df = df.groupby("ts", as_index=False).agg({"evtval_count": ["mean"]})

    data_x = df["ts"]
    data_y = df["evtval_count"]["mean"]

    ax.plot(data_x, data_y)
    ax.set_title("Datapoints Salvaged (Out of 512) For Each AMR TS")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number Of Datapoints (= Ranks) Parseable")
    # fig.show()
    fig.savefig("{}/taskflow_nbrcnt.pdf".format(plot_dir), dpi=300)


def get_data(df, evt, col):
    df = df[df["evtname"] == evt]
    data_x = df["ts"]
    data_y = df[col]
    return data_x, data_y


def plot_event(event_name, df, plot_dir, plot_tail=False, save=False):
    fig, ax = plt.subplots(1, 1)
    cm = plt.cm.get_cmap("tab20c")

    dx, dy = get_data(df, event_name, "evtval_mean")
    ax.plot(dx, dy, color=cm(0), label="Mean ({})".format(event_name))

    dx, dy = get_data(df, event_name, "evtval_percentile_50")
    ax.plot(dx, dy, "--", color=cm(4), label="50th %-ile ({})".format(event_name))

    dx, dy = get_data(df, event_name, "evtval_percentile_75")
    ax.plot(dx, dy, "--", color=cm(8), label="75th %-ile ({})".format(event_name))

    if plot_tail:
        dx, dy = get_data(df, event_name, "evtval_percentile_99")
        ax.plot(dx, dy, "--", color=cm(12), label="99th %-ile ({})".format(event_name))

    ax.set_title("Statistics for Event {}".format(event_name))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")

    ax.legend()
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f}s".format(x / 1e6))

    ax.set_xlim([4000, 6000])

    event_key = event_name.lower()
    event_key = "{}_zoomed".format(event_name.lower())

    plot_fname = None
    if plot_tail:
        plot_fname = "taskflow_{}_w99.pdf".format(event_key)
    else:
        plot_fname = "taskflow_{}_wo99.pdf".format(event_key)

    fig.tight_layout()
    if save:
        fig.savefig("{}/{}".format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()


def plot_all_events(df, plot_dir):
    for event in ["AR1", "AR2", "AR3", "SR"]:
        plot_event(event, df, plot_dir, plot_tail=False, save=True)
        plot_event(event, df, plot_dir, plot_tail=True, save=True)


def plot_amr_log(log_df, plot_dir, save=False):
    print(log_df)

    fig, ax = plt.subplots(1, 1)

    key_y = "wtime_step_other"
    label_y = "Walltime (Non-AMR/LB)"
    data_x = log_df["cycle"]
    data_y = log_df[key_y]
    ax.plot(data_x, data_y, label=label_y)

    key_y = "wtime_step_amr"
    label_y = "Walltime (AMR/LB)"
    data_x = log_df["cycle"]
    data_y = log_df[key_y]
    ax.plot(data_x, data_y, label=label_y)

    ax.set_title("Wall Time for AMR Run (512 Timesteps)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Walltime (seconds)")

    ax.legend()

    plot_fname = "amr_steptimes.pdf"

    # ax.set_xlim([3750, 4250])
    # plot_fname = 'amr_steptimes_zoomed.pdf'

    fig.tight_layout()

    if save:
        fig.savefig("{}/{}".format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()


def calc_amr_log_stats(log_df):
    def calc_key_stats(key):
        print("Analyzing {}".format(key))
        data_y = log_df[key]
        med_val = np.median(data_y)
        sum_val = data_y.sum()
        print("Median: {:.2f}, Sum: {:.2f}".format(med_val, sum_val))

        first_half = sum([i for i in data_y if i < med_val])
        second_half = sum([i for i in data_y if i > med_val])

        print(
            "Sums: {:.1f}/{:.1f} (First 50%, Last 50%)".format(first_half, second_half)
        )

    data_y = log_df["wtime_step_other"]
    calc_key_stats("wtime_step_other")
    calc_key_stats("wtime_step_amr")


def plot_amr_log_distrib(log_df, plot_dir, save=False):
    fig, ax = plt.subplots(1, 1)

    data_y = log_df["wtime_step_other"]
    plt.hist(
        data_y,
        bins=100,
        density=0,
        histtype="step",
        cumulative=True,
        label="Non-AMR/LB (Cumul.)",
    )
    # plt.hist(data_y, bins=100, density=0, histtype='step', cumulative=False, label='Non-AMR/LB')
    data_y = log_df["wtime_step_amr"]
    plt.hist(
        data_y,
        bins=100,
        density=0,
        histtype="step",
        cumulative=True,
        label="AMR/LB (Cumul)",
    )
    # plt.hist(data_y, bins=100, density=0, histtype='step', cumulative=False, label='AMR/LB')

    ax.legend()

    noncum_profile = True
    zoomed_profile = False
    save = True

    if noncum_profile:
        ax.set_xlim([0, 3])
        ax.set_title("Wall Time for AMR Run (512 Timesteps)")
        ax.set_xlabel("Walltime (seconds)")
        ax.set_ylabel("Num Times")
        plot_fname = "amr_steptimes_distrib_noncumul.pdf"
    else:
        ax.set_title("Wall Time for AMR Run (512 Timesteps)")
        ax.set_xlabel("Walltime (seconds)")
        ax.set_ylabel("Number Of Timesteps > X")

        ax.yaxis.set_major_formatter(lambda x, pos: max(round(30000 * (1 - x)), 0))

        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        if zoomed_profile:
            ax.set_ylim([0.99, 1.001])
            plot_fname = "amr_steptimes_distrib.pdf"
            ax.yaxis.set_major_locator(MultipleLocator(0.002))
            ax.yaxis.set_minor_locator(MultipleLocator(0.001))
        else:
            plot_fname = "amr_steptimes_distrib.pdf"
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.grid(visible=True, which="major", color="#999")
    plt.grid(visible=True, which="minor", color="#ddd")
    fig.tight_layout()

    if save:
        fig.savefig("{}/{}".format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()


def plot_amr_comp(all_dfs, plot_dir, save=False):
    fig, ax = plt.subplots(1, 1)

    cm = plt.cm.get_cmap("Set1")

    for idx, df in enumerate(all_dfs):
        data_x = df["cycle"]
        data_y1 = df["wtime_step_other"]
        data_y2 = df["wtime_step_amr"]

        label_1 = "Run{}-Kernel".format(idx)
        label_2 = "Run{}-AMR".format(idx)
        ax.plot(data_x, data_y1.cumsum(), label=label_1, color=cm(idx))
        ax.plot(data_x, data_y2.cumsum(), label=label_2, linestyle="--", color=cm(idx))

    ax.set_title("AMR Runs (512 Ranks) Phasewise Cumul. Times")
    ax.set_xlabel("Timestep")
    ax.set_xlabel("Total Time (seconds)")

    ax.legend()
    plt.grid(visible=True, which="major", color="#999")
    plt.grid(visible=True, which="minor", color="#ddd")
    fig.tight_layout()

    plot_fname = "amr_steptimes_comp.pdf"
    plot_fname = "amr_steptimes_comp_zoomed.pdf"
    ax.set_xlim([0000, 10000])

    # save = True

    if save:
        fig.savefig("{}/{}".format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()
    pass


def run_plot_amr_comp():
    plot_dir = "figures_bigrun"
    log_dirs = [
        "/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log.times.csv",
        "/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log2.csv",
        "/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log3.csv",
        "/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log4.csv",
    ]

    log_dfs = map(pd.read_csv, log_dirs)
    plot_amr_comp(log_dfs, plot_dir, save=False)


def plot_profile():
    pass


def run_profile():
    df_path = "/Users/schwifty/Repos/amr-data/20220524-phase-analysis/profile.log.csv"
    df = pd.read_csv(df_path)
    df = df.astype({"rank": "int32", "event": str, "timepct": float})
    events = df["event"].unique()

    fig, ax = plt.subplots(1, 1)

    for event in events:
        dfe = df[df["event"] == event]
        data_x = dfe["rank"]
        data_y = dfe["timepct"]
        print(data_x)
        print(data_y)
        ax.plot(dfe["rank"], dfe["timepct"], label=event)

    ax.set_title("Function-Wise Times (Pct Of Process Time)")
    ax.set_xlabel("Rank Index")
    ax.set_ylabel("Time Taken (%)")
    ax.legend()
    # fig.tight_layout()
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.0f}%".format(x))
    fig.show()
    plot_dir = "figures_bigrun"
    plot_fname = "amr_profile_phases.pdf"
    fig.savefig("{}/{}".format(plot_dir, plot_fname), dpi=300)


def strtols(list_str):
    ls = list_str.strip("[]").split(",")
    ls = np.array([float(i) for i in ls])
    ls /= 1e6
    return ls


def plot_timestep(df_ts, df_log, fpath):
    get_data_y = lambda x: strtols(df_ts[df_ts["evtname"] == x]["evtval"].iloc[0])

    timestep = df_ts["ts"].unique()[0]
    print(timestep)
    nranks = 512

    cmap = plt.cm.get_cmap("tab20b")

    fig, ax = plt.subplots(1, 1)
    data_x = range(nranks)

    data_pt = df_log["wtime_step_other"] + df_log["wtime_step_amr"]
    ax.plot(
        [0, nranks - 1],
        [data_pt, data_pt],
        linestyle="--",
        label="Internal (Total)",
        color=cmap(0),
    )

    data_y = get_data_y("TIME_CLASSIFIEDPHASES") - get_data_y("AR3_UMBT")
    ax.plot(data_x, data_y, label="TauClassified", color=cmap(1))

    data_y = get_data_y("AR1")
    ax.plot(data_x, data_y, label="Tau AR1", color=cmap(8))
    data_y = get_data_y("AR2")
    ax.plot(data_x, data_y, label="Tau AR2", color=cmap(9))

    data_pt = df_log["wtime_step_amr"]
    ax.plot(
        [0, nranks - 1],
        [data_pt, data_pt],
        linestyle="--",
        label="Internal (AMR)",
        color=cmap(17),
    )
    data_y = get_data_y("AR3")
    ax.plot(data_x, data_y, linestyle=":", label="Tau AR3", color=cmap(13))
    data_y = get_data_y("AR3_UMBT")
    ax.plot(data_x, data_y, label="Tau AR3_AllGather", color=cmap(12))

    #  data_y = get_data_y('TIME_FROMPREVEND')
    #  ax.plot(data_x, data_y, label='Tau StepTimeFromPrevEnd')

    ax.legend(ncol=3, prop={"size": 10})

    ax.set_xlabel("Rank #")
    ax.set_ylabel("Time (s)")
    ax.set_title("Rankwise Time Breakdown For Step {}".format(timestep))
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.2f}".format(x))
    fig.tight_layout()
    fig.savefig("{}/plot_step/plot_step{}.pdf".format(fpath, timestep), dpi=300)


def plot_logstats(df_logstats, plot_dir: str) -> None:
    fig, ax = plt.subplots(1, 1)

    data_x = range(len(df_logstats))

    dy_compute = np.array(df_logstats["wtime_step_other"])
    dy_amr = np.array(df_logstats["wtime_step_amr"])

    print("Total Phases: {:.0f}/{:.0f}".format(sum(dy_compute), sum(dy_amr)))

    ax.plot(data_x, dy_compute, label="Internal (Compute+BC)")
    #  ax.plot(data_x, dy_amr, label='Internal (AMR)')
    #  ax.plot(data_x, dy_compute + dy_amr, label='Internal (Total)')

    ax.set_xlim([9500, 10500])

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")

    ax.set_title("Time Taken For Two Phases As Per Internal Timers")
    ax.legend()
    fig.savefig("{}/timeline_a_zoomed.pdf".format(plot_dir), dpi=300)


def get_all_and_aggr(df, key, aggr_f):
    df = df[df["evtname"] == key]
    mapobj = map(lambda x: aggr_f(strtols(x)), df["evtval"])
    mapobj = list(mapobj)
    print(len(mapobj))
    return np.array(mapobj)


def plot_umbt_stats(df_phases, df_log, plot_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    get_data_y = lambda x: strtols(
        df_phases[df_phases["evtname"] == x]["evtval"].iloc[0]
    )

    min5 = lambda x: sorted(x)[5]

    data_y1 = get_all_and_aggr(df_phases, "AR3", max)
    data_y1a = get_all_and_aggr(df_phases, "AR3_UMBT", max)
    data_y1b = get_all_and_aggr(df_phases, "AR3_UMBT", min)
    data_y1c = get_all_and_aggr(df_phases, "AR3_UMBT", min5)
    data_y1d = get_all_and_aggr(df_phases, "AR3_UMBT", np.median)
    data_y2 = df_log["wtime_step_amr"]
    data_x = range(len(data_y2))

    common_len = min(
        [len(i) for i in [data_y1, data_y1a, data_y1b, data_y1c, data_y1d, data_y2]]
    )
    data_y1 = data_y1[:common_len]
    data_y1a = data_y1a[:common_len]
    data_y1b = data_y1b[:common_len]
    data_y1c = data_y1c[:common_len]
    data_y1d = data_y1d[:common_len]
    data_y2 = data_y2[:common_len]

    print(data_y1)
    print(data_y2)
    assert len(data_y1) == len(data_y2)
    assert len(data_y1a) == len(data_y2)

    ax.plot(data_x, data_y1, label="TAU (AR3)")
    ax.plot(data_x, data_y1a, label="TAU (AR3_UMBT)")
    ax.plot(data_x, data_y2, label="Internal (AMR)")

    ax.legend()
    ax.set_title("Times For Different AMR Phases")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.2f}s".format(x))

    fig.tight_layout()
    fig.savefig("{}/amr_stats.pdf".format(plot_dir), dpi=300)

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y1a / data_y1, label="TAU_UMBT/TAU_AR3")
    ax.plot(data_x, data_y1a / data_y2, label="TAU_UMBT/INT_AMR")

    ax.legend()
    ax.set_title("Ratio Of Collective Time To Total AMR-LB Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time Ratio")
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f}%".format(x * 100))

    fig.tight_layout()
    fig.savefig("{}/amr_timerat.pdf".format(plot_dir), dpi=300)

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y1a / data_y1, label="TAU_UMBT/TAU_AR3")
    ax.plot(data_x, data_y1a / data_y2, label="TAU_UMBT/INT_AMR")

    ax.legend()
    ax.set_title("Ratio Of Collective Time To Total AMR-LB Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time Ratio")
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.1f}%".format(x * 100))

    ax.set_ylim([0, 1.5])

    fig.tight_layout()
    fig.savefig("{}/amr_timerat_clipped.pdf".format(plot_dir), dpi=300)

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_x, data_y1a, label="$AG_{NO}$:Max", alpha=0.7)
    ax.plot(data_x, data_y1d, label="$AG_{NO}$:Median", alpha=0.7)
    ax.plot(data_x, data_y1c, label="$AG_{NO}$:Min5", alpha=0.7)
    ax.plot(data_x, data_y1b, label="$AG_{NO}$:Min", alpha=0.7)

    ax.legend()
    ax.set_title("Min And Max Collective Times Across All Ranks")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.2f}s".format(x))

    fig.tight_layout()
    fig.savefig("{}/amr_umbt_min_max.pdf".format(plot_dir), dpi=300)

    fig, ax = plt.subplots(1, 1)
    data_y1amb = data_y1a - data_y1b
    data_y1amc = data_y1a - data_y1c
    data_y1amd = data_y1a - data_y1d

    ax.plot(data_x, data_y1amb, label="TAU_UMBT_MAX-MIN")
    ax.plot(data_x, data_y1amc, label="TAU_UMBT_MAX-MIN5")
    ax.plot(data_x, data_y1amd, label="TAU_UMBT_MAX-MED")

    ax.legend()
    ax.set_title("Min And Max Collective Times Across All Ranks")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(lambda x, pos: "{:.2f}s".format(x))

    fig.tight_layout()
    fig.savefig("{}/amr_umbt_minmax_delta.pdf".format(plot_dir), dpi=300)

    print("Sum UMBT_MAX: {:.0f}".format(sum(data_y1a)))
    print("Sum UMBT_MAX-MIN: {:.0f}".format(sum(data_y1amb)))
    print("Sum UMBT_MAX-MIN5: {:.0f}".format(sum(data_y1amc)))
    print("Sum UMBT_MAX-MED: {:.0f}".format(sum(data_y1amd)))


def plot_umbt_rankgrid(df_phases, imevent, plot_dir, cached=False):
    def sort_xargs(ls):
        ls_widx = [(ls[i], i) for i in range(len(ls))]
        ls_widx = sorted(ls_widx)
        ls_idx = [i[1] for i in ls_widx]
        #  ls_idx = np.array(ls_idx)
        return ls_idx

    CACHE_FNAME = ".rankgrid.{}".format(imevent)
    data_ranks = None

    if not cached:
        data_ranks = get_all_and_aggr(df_phases, imevent, sort_xargs)
        with open(CACHE_FNAME, "wb+") as f:
            f.write(pickle.dumps(data_ranks))
    else:
        with open(CACHE_FNAME, "rb") as f:
            data_ranks = pickle.loads(f.read())

    data_ranks = list(data_ranks[:-1])
    data_ranks = np.vstack(data_ranks)
    #  print(data_ranks.shape)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data_ranks, aspect="auto", cmap="plasma")

    ax.set_title("Rank Order For Event {}".format(imevent))
    ax.set_ylabel("Timestep")
    ax.xaxis.set_ticks([])
    ax.set_xlabel("Ranks In Increasing Order Of Phase Time")

    plt.subplots_adjust(left=0.15, right=0.8)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    fig.colorbar(im, cax=cax)

    fig.savefig("{}/umbt_rankgrid_{}.pdf".format(plot_dir, imevent.lower()), dpi=600)


def plot_umbt_rankgrid_wcompare(df_phases, df_log, imevent, plot_dir, cached=False):
    def sort_xargs(ls):
        ls_widx = [(ls[i], i) for i in range(len(ls))]
        ls_widx = sorted(ls_widx)
        ls_idx = [i[1] for i in ls_widx]
        #  ls_idx = np.array(ls_idx)
        return ls_idx

    CACHE_FNAME = ".phasetimegrid.{}".format(imevent)
    data_ranks = None

    if not cached:
        data_ranks = get_all_and_aggr(df_phases, imevent, lambda x: x)
        with open(CACHE_FNAME, "wb+") as f:
            f.write(pickle.dumps(data_ranks))
    else:
        with open(CACHE_FNAME, "rb") as f:
            data_ranks = pickle.loads(f.read())

    data_times = list(data_ranks[:-1])
    data_times = np.vstack(data_times)
    #  print(data_ranks.shape)

    def smoothen(data):
        kernel_size = 20
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_20 = np.convolve(data, kernel, mode="same")
        return data_convolved_20

    data_time_med = np.median(data_times, axis=1)
    data_time_max = np.max(data_times, axis=1)
    data_time_log = df_log["wtime_step_other"]

    print(data_time_med.shape)

    #  fig, axes = plt.subplots(1, 4, gridspec_kw={"width_ratios": [4, 1, 1, 1]})
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [4, 1, 1]})

    ax_im = axes[0]
    im = ax_im.imshow(data_times, aspect="auto", cmap="plasma")

    num_ts = data_times.shape[0]
    data_y = range(num_ts)

    axes[1].plot(data_time_med, data_y)
    axes[2].plot(data_time_max, data_y)
    #  axes[3].plot(data_time_log[:num_ts], data_y)

    # ax.set_title('Rank Order For Event {}'.format(imevent))
    # ax.set_ylabel('Timestep')
    # ax.xaxis.set_ticks([])
    # ax.set_xlabel('Ranks In Increasing Order Of Phase Time')
    axes[0].xaxis.set_ticks([])
    axes[0].set_title("Rankwise Phase Times")

    #  axes[1].xaxis.set_ticks([])
    axes[1].yaxis.set_ticks([])
    axes[1].set_title("TauMed")
    axes[1].set_ylim([num_ts, 0])

    #  axes[2].xaxis.set_ticks([])
    axes[2].yaxis.set_ticks([])
    axes[2].set_title("TauMax")
    axes[2].set_ylim([num_ts, 0])

    #  axes[3].yaxis.set_ticks([])
    #  axes[3].set_ylim([num_ts, 0])

    fig.suptitle("Time Distributions For Event {}".format(imevent))
    # fig.supxlabel('Something')
    fig.supylabel("Timesteps")

    # plt.subplots_adjust(left=0.15, right=0.8)
    plt.subplots_adjust(wspace=0.03, left=0.15)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # fig.colorbar(im, cax=cax)
    fig.colorbar(im, ax=axes[-1])

    save = True
    if save:
        fig.savefig(
            "{}/umbt_phasetimegrid_{}.pdf".format(plot_dir, imevent.lower()), dpi=600
        )
    else:
        fig.show()


def plot_umbt_rankgrid_wcompare_amr(df_phases, df_log, plot_dir, cached=False):
    def sort_xargs(ls):
        ls_widx = [(ls[i], i) for i in range(len(ls))]
        ls_widx = sorted(ls_widx)
        ls_idx = [i[1] for i in ls_widx]
        #  ls_idx = np.array(ls_idx)
        return ls_idx

    CACHE_FNAME = ".phasetimegrid.amr"
    data_ranks = None

    if not cached:
        data_ranks = get_all_and_aggr(df_phases, "AR3", lambda x: x)
        with open(CACHE_FNAME, "wb+") as f:
            f.write(pickle.dumps(data_ranks))
    else:
        with open(CACHE_FNAME, "rb") as f:
            data_ranks = pickle.loads(f.read())

    data_times = list(data_ranks[:-1])
    data_times = np.vstack(data_times)
    #  print(data_ranks.shape)

    def smoothen(data):
        kernel_size = 20
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_20 = np.convolve(data, kernel, mode="same")
        return data_convolved_20

    data_time_med = np.median(data_times, axis=1)
    data_time_max = np.max(data_times, axis=1)
    data_time_log_comp = df_log["wtime_step_other"]
    data_time_log_amr = df_log["wtime_step_amr"]

    print(data_time_med.shape)

    fig, axes = plt.subplots(1, 5, gridspec_kw={"width_ratios": [4, 1, 1, 1, 1]})

    ax_im = axes[0]
    im = ax_im.imshow(data_times, aspect="auto", cmap="plasma")

    num_ts = data_times.shape[0]
    data_y = range(num_ts)

    axes[1].plot(data_time_med, data_y)
    axes[2].plot(data_time_max, data_y)
    axes[3].plot(data_time_log_comp[:num_ts], data_y)
    axes[4].plot(data_time_log_amr[:num_ts], data_y)

    # ax.set_title('Rank Order For Event {}'.format(imevent))
    # ax.set_ylabel('Timestep')
    # ax.xaxis.set_ticks([])
    # ax.set_xlabel('Ranks In Increasing Order Of Phase Time')
    axes[0].xaxis.set_ticks([])
    axes[0].set_title("Rankwise Phase Times")

    #  axes[1].xaxis.set_ticks([])
    axes[1].yaxis.set_ticks([])
    axes[1].set_title("T_Med")
    axes[1].set_ylim([num_ts, 0])

    #  axes[2].xaxis.set_ticks([])
    axes[2].yaxis.set_ticks([])
    axes[2].set_title("T_Max")
    axes[2].set_ylim([num_ts, 0])

    axes[3].yaxis.set_ticks([])
    axes[3].set_title("I_Comp")
    axes[3].set_ylim([num_ts, 0])

    axes[4].yaxis.set_ticks([])
    axes[4].set_title("I_Amr")
    axes[4].set_ylim([num_ts, 0])

    fig.suptitle("Time Distributions For AR3=AMR")
    # fig.supxlabel('Something')
    fig.supylabel("Timesteps")

    # plt.subplots_adjust(left=0.15, right=0.8)
    plt.subplots_adjust(wspace=0.03, left=0.15)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # fig.colorbar(im, cax=cax)
    fig.colorbar(im, ax=axes[-1])

    save = True
    if save:
        fig.savefig("{}/umbt_phasetimegrid_amr.pdf".format(plot_dir), dpi=600)
    else:
        fig.show()


def plot_umbt_rankgrid_wcompare_nonamr(df_phases, df_log, plot_dir, cached=False):
    def sort_xargs(ls):
        ls_widx = [(ls[i], i) for i in range(len(ls))]
        ls_widx = sorted(ls_widx)
        ls_idx = [i[1] for i in ls_widx]
        #  ls_idx = np.array(ls_idx)
        return ls_idx

    CACHE_FNAME = ".phasetimegrid.nonamr"
    data_ranks = None

    if not cached:
        data = map(
            lambda key: get_all_and_aggr(df_phases, key, lambda x: x),
            ["AR1", "AR2", "SR"],
        )
        #  IPython.embed()
        data = list(data)
        dims = [i.shape for i in data]
        dim_min = min([i[0] for i in dims]) - 1
        print("Dims: ", dims)
        print("Dim Min: ", dim_min)

        data_minned = [i[:dim_min] for i in data]
        data_ranks = np.sum(data_minned, axis=0)

        print("Data Shape: ", data_ranks.shape)

        with open(CACHE_FNAME, "wb+") as f:
            f.write(pickle.dumps(data_ranks))
    else:
        with open(CACHE_FNAME, "rb") as f:
            data_ranks = pickle.loads(f.read())

    data_times = list(data_ranks[:-1])
    data_times = np.vstack(data_times)
    print(data_ranks.shape)

    def smoothen(data):
        kernel_size = 20
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_20 = np.convolve(data, kernel, mode="same")
        return data_convolved_20

    data_time_med = np.median(data_times, axis=1)
    data_time_max = np.max(data_times, axis=1)
    data_time_log_comp = df_log["wtime_step_other"]
    data_time_log_amr = df_log["wtime_step_amr"]

    print(data_time_med.shape)

    fig, axes = plt.subplots(1, 5, gridspec_kw={"width_ratios": [4, 1, 1, 1, 1]})

    ax_im = axes[0]
    im = ax_im.imshow(data_times, aspect="auto", cmap="plasma")

    num_ts = data_times.shape[0]
    data_y = range(num_ts)

    axes[1].plot(data_time_med, data_y)
    axes[2].plot(data_time_max, data_y)
    axes[3].plot(data_time_log_comp[:num_ts], data_y)
    axes[4].plot(data_time_log_amr[:num_ts], data_y)

    axes[0].xaxis.set_ticks([])
    axes[0].set_title("Rankwise Phase Times")

    axes[1].yaxis.set_ticks([])
    axes[1].set_title("T_Med")
    axes[1].set_ylim([num_ts, 0])

    axes[2].yaxis.set_ticks([])
    axes[2].set_title("T_Max")
    axes[2].set_ylim([num_ts, 0])

    axes[3].yaxis.set_ticks([])
    axes[3].set_title("I_Comp")
    axes[3].set_ylim([num_ts, 0])

    axes[4].yaxis.set_ticks([])
    axes[4].set_title("I_AMR")
    axes[4].set_ylim([num_ts, 0])

    fig.suptitle("Time Distributions For AR1+AR2+SR=NonAMR")
    fig.supylabel("Timesteps")

    plt.subplots_adjust(wspace=0.03, left=0.15)
    fig.colorbar(im, ax=axes[-1])

    save = True
    if save:
        fig.savefig("{}/umbt_phasetimegrid_nonamr.pdf".format(plot_dir), dpi=600)
    else:
        fig.show()


def run_plot_timestep():
    trace_dir = "/mnt/ltio/parthenon-topo/profile6.wtau"
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    plot_dir = "/users/ankushj/repos/amr/scripts/tau_analysis/figures"
    plot_dir = "figures/20220809"
    ts_to_plot = 1

    cached = False

    df_phases = None

    if not cached:
        df_phases = pd.read_csv("{}/aggregate.csv".format(trace_dir))

    df_log = pd.read_csv("{}/run/log.txt.csv".format(trace_dir)).astype({"cycle": int})

    #  plot_umbt_rankgrid(df_phases, "AR1", plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare(df_phases, df_log, "AR1", plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare(df_phases, df_log, "AR2", plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare(df_phases, df_log, "SR", plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare(df_phases, df_log, "AR3", plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare(df_phases, df_log, "AR3_UMBT", plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare_nonamr(df_phases, df_log, plot_dir, cached=cached)
    #  plot_umbt_rankgrid_wcompare_amr(df_phases, df_log, plot_dir, cached=cached)
    #  plot_umbt_rankgrid(df_phases, "AR2", plot_dir, cached=cached)
    #  plot_umbt_rankgrid(df_phases, "AR3", plot_dir, cached=cached)
    #  plot_umbt_rankgrid(df_phases, "AR3_UMBT", plot_dir, cached=cached)
    plot_umbt_stats(df_phases, df_log, plot_dir)
    return

    ts_selected = df_log[df_log["wtime_step_other"] > 0.6]["cycle"]
    ts_to_plot = []
    for ts in ts_selected:
        ts_to_plot.append(ts - 1)
        ts_to_plot.append(ts)
        ts_to_plot.append(ts + 1)
    print(ts_to_plot)

    for ts in ts_to_plot:
        print(ts)
        df_ts = df_phases[df_phases["ts"] == ts]
        df_logts = df_log[df_log["cycle"] == ts]
        print(df_ts)
        print(df_logts)
        plot_timestep(df_ts, df_logts, plot_dir)

    #  plot_logstats(df_log, plot_dir)
    return


""" Input: trace/taskaggr.csv 
Output: XX
"""


def run_plot_aggr(trace_dir: str, plot_dir):
    tr = TraceOps(trace_dir)
    aggr_df_path = "{}/trace/taskaggr.csv".format(trace_dir)
    aggr_df = pd.read_csv(aggr_df_path)
    print(aggr_df)

    def get_data(key):
        row = aggr_df[aggr_df["evtname"] == key]["evtval"].iloc[0]
        row = np.array([int(i) for i in row.split(",")], dtype=np.int64)
        return row

    def plot_total_phasetimes():
        data_ar1 = get_data("AR1")
        data_ar2 = get_data("AR2")
        data_sr = get_data("SR")
        data_ar3 = get_data("AR3")
        data_ar3u = get_data("AR3_UMBT")

        nranks = 512
        data_x = list(range(nranks))
        fig, ax = plt.subplots(1, 1)

        #  ipdb.set_trace()

        ax.plot(data_x, data_ar1, label="$FC_{CN}$")
        ax.plot(data_x, data_sr, label="$BC_{NO}$")
        ax.plot(data_x, data_ar2, label="$FD_{CO}$")
        ax.plot(data_x, data_ar3u, label="$AG_{NO}$")
        ax.plot(data_x, data_ar3 - data_ar3u, label="$LB_{NO}$")

        ax.set_xlabel("Rank ID")
        ax.set_ylabel("Total Time (s)")
        ax.set_title("Total Time For Each Phase/Rank")

        ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}s'.format(x/1e6))
        ax.legend(bbox_to_anchor=(-0.15, 1.08), loc="lower left", ncol=5)
        fig.tight_layout()

        plot_dest = "{}/phases.aggr.pdf".format(plot_dir)
        print("Saving plot: {}".format(plot_dest))
        fig.savefig(plot_dest, dpi=300)

    def plot_lbvsmsgcnt():
        msg_mat_lb = tr.multimat("msgcnt:LoadBalancing")
        msg_mat_bc = tr.multimat("msgcnt:BoundaryComm")
        lbmsg_rwtotals = np.sum(msg_mat_lb, axis=0)
        bcmsg_rwtotals = np.sum(msg_mat_bc, axis=0)

        data_lb = get_data("AR3") - get_data("AR3_UMBT")

        nranks = 512
        data_x = list(range(nranks))

        fig, ax = plt.subplots(1, 1)
        ax2 = ax.twinx()

        ax.plot(data_x, lbmsg_rwtotals, label='Message Count - LB')
        ax.plot(data_x, bcmsg_rwtotals, label='Message Count - BC')
        ax2.plot(data_x, data_lb, label='Time', color='orange')

        ax.set_xlabel("Rank ID")
        ax.set_ylabel("Message Count")
        ax2.set_ylabel("Time AR3_LB (s)")

        ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}K'.format(x/1e3))
        ax2.yaxis.set_major_formatter(lambda x, pos: '{:.0f} s'.format(x/1e6))

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax2.set_ylim([0, ax2.get_ylim()[1]])

        ax.legend()

        plot_dest = "{}/ar3_vs_msgcnt.pdf".format(plot_dir)
        fig.tight_layout()
        fig.savefig(plot_dest, dpi=300)

    def plot_lb_someranks():
        ranks_to_plot = [367, 368]

        lb_mat = tr.multimat("tau:AR3-AR3_UMBT")
        data_x = list(range(lb_mat.shape[0]))

        fig, ax = plt.subplots(1, 1)

        for r in ranks_to_plot:
            data_ry = lb_mat[:, r]
            ax.plot(data_x, data_ry, label='Rank {}'.format(r))

        ax.legend()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Time ms')

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.xaxis.set_major_formatter(lambda x, pos: '{:0.0f} ms'.format(x/1e3))

        plot_dest = "{}/lb_rankwise.pdf".format(plot_dir)
        fig.tight_layout()
        fig.savefig(plot_dest, dpi=300)


    plot_total_phasetimes()
    #  plot_lbvsmsgcnt()
    #  plot_lb_someranks()


def run_analyze(trace_dir: str):
    tr = TraceReader(trace_dir)
    all_events = ["AR1", "AR2", "SR", "AR3", "AR3_UMBT"]
    for evt in all_events:
        mat = tr.get_tau_event(evt)
        mat_ts_mean = np.mean(mat, axis=1)
        print(mat_ts_mean.shape)
        mat_ts_sum = np.sum(mat_ts_mean)
        print("{}: {:.1f}s".format(evt, mat_ts_sum / 1e6))


def run_plot():
    trace_dir = "/mnt/ltio/parthenon-topo/profile9"
    # aggr_fpath = '/Users/schwifty/repos/amr-data/20220517-phase-analysis/aggregate.csv'
    # df = pd.read_csv(aggr_fpath)
    plot_init()
    plot_dir = "figures/20220811-profile9"
    # # plot_neighbors(df, plot_dir)
    # plot_all_events(df, plot_dir)

    # phoebus_log = '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log.times.csv'
    # phoebus_log2 = '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log2.csv'
    phoebus_log = "{}/run/log.txt.csv".format(trace_dir)
    #  log_df = pd.read_csv(phoebus_log)
    # log_df2 = pd.read_csv(phoebus_log2)
    #  plot_amr_log(log_df, plot_dir, save=True)
    #  plot_amr_log_distrib(log_df, plot_dir, save=True)
    #  calc_amr_log_stats(log_df)
    #  run_plot_amr_comp()
    #  run_profile()
    #  run_plot_timestep()
    #  run_analyze(trace_dir)
    run_plot_aggr(trace_dir, plot_dir)


if __name__ == "__main__":
    run_plot()
