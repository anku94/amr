import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common import plot_init, plot_init_big, PlotSaver, profile_label_map


global trace_dir_fmt
trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"


# hardcoded for profile40 to profile43
def get_rankhour_comparison_blastwave():
    comp_fd = np.array([2001, 2010, 2001, 2004])
    comp_cf = np.array([1442, 1442, 1442, 1444])
    phase_data = {
        "app": np.array([9093, 8125, 8117, 7979]),
        "comp": comp_fd + comp_cf,
        "sync": np.array([646, 237, 334, 282]),
        "lb": np.array([73, 79, 72, 76]),
        "send": np.array([468, 635, 476, 478]),
        "recv": np.array([384, 315, 300, 286]),
    }

    keys_all = ["comp", "sync", "lb", "send", "recv"]
    time_def_phases = np.sum([phase_data[k] for k in keys_all], axis=0)

    phase_data["other"] = phase_data["app"] - time_def_phases
    phase_data

    return phase_data


def plot_rankhour_comparison_simple(trace_names: list[str]) -> None:
    n_traces = len(trace_names)
    width = 0.45

    phase_data_summ = get_rankhour_comparison_blastwave()

    trace_labels = ["Baseline", "LPT", "Contiguous-DP", "Contiguous-DP++"]

    keys_to_plot = ["comp", "send", "recv", "sync", "lb"]
    keys_to_plot = ["comp", "sync", "send", "recv", "lb"]
    key_labels = ["Compute", "Global Barrier", "MPI Send", "MPI Recv", "LoadBalancing"]
    bottom = np.zeros(n_traces)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    for idx, k in enumerate(keys_to_plot):
        data_x = np.arange(n_traces)
        data_y = phase_data_summ[k]

        label = k[0].upper() + k[1:]
        label = key_labels[idx]
        ax.bar(data_x, data_y, bottom=bottom, zorder=2, width=width, label=label)
        bottom += data_y

    p = ax.bar(
        data_x,
        phase_data_summ["other"],
        bottom=bottom,
        zorder=2,
        width=width,
        label="Other",
        color="#999",
    )

    ax.bar_label(
        p,
        fmt="{:.0f} s",
        rotation="horizontal",
        label=phase_data_summ["app"],
        fontsize=14,
        padding=4,
    )

    ax.set_xticks(data_x)
    ax.set_xticklabels(trace_labels)
    ax.set_ylabel("Runtime (s)")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} s".format(x))
    )

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(400))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 12000])

    ax.legend(loc="upper right", ncol=3, fontsize=13)

    fig.tight_layout()

    plot_fname = "pprof_rh_simple_blastwave"
    PlotSaver.save(fig, "", None, plot_fname)
    pass


def plot_rankhour_comparison_simple_build(trace_names: list[str]) -> None:
    n_traces = len(trace_names)
    width = 0.45

    phase_data_summ = get_rankhour_comparison_blastwave()

    trace_labels = ["Baseline", "LPT", "Contiguous-DP", "Contiguous-DP++"]

    keys_to_plot = ["comp", "send", "recv", "sync", "lb"]
    keys_to_plot = ["comp", "sync", "send", "recv", "lb"]
    key_labels = ["Compute", "Global Barrier", "MPI Send", "MPI Recv", "LoadBalancing"]
    bottom = np.zeros(n_traces)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    data_x = np.arange(n_traces)

    ax.set_xlim(-0.3975, 3.3975)

    ax.set_title("Runtime Evolution in Blast Wave (30k timesteps)", fontsize=18)
    ax.set_xticks(data_x[:n_traces])
    ax.set_xticklabels(trace_labels[:n_traces])
    ax.set_ylabel("Runtime (s)")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "{:.0f} s".format(x))
    )

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(400))
    ax.yaxis.grid(which="major", visible=True, color="#bbb", zorder=0)
    ax.yaxis.grid(which="minor", visible=True, color="#ddd", zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim([0, 12000])

    fig.tight_layout()

    fig.subplots_adjust(right=0.93)

    for idx, k in enumerate(keys_to_plot):
        data_y = phase_data_summ[k][:n_traces]

        label = k[0].upper() + k[1:]
        label = key_labels[idx]
        ax.bar(data_x, data_y, bottom=bottom, zorder=2, width=width, label=label)
        bottom += data_y

    p = ax.bar(
        data_x,
        phase_data_summ["other"][:n_traces],
        bottom=bottom,
        zorder=2,
        width=width,
        label="Other",
        color="#999",
    )

    ax.bar_label(
        p,
        fmt="{:.0f} s",
        rotation="horizontal",
        label=phase_data_summ["app"],
        fontsize=14,
        padding=4,
    )

    ax.legend(loc="upper right", ncol=3, fontsize=13)

    plot_fname = f"pprof_rh_simple_blastwave_{len(trace_names)}"
    PlotSaver.save(fig, "", None, plot_fname)


def run():
    global trace_dir_fmt
    traces = ["profile40", "profile41", "profile42", "profile43"]
    trace_dirs = list(map(lambda x: trace_dir_fmt.format(x), traces))
    trace_names = traces
    print(trace_dirs)
    plot_rankhour_comparison_simple(traces)
    plot_rankhour_comparison_simple_build(traces[:1])
    plot_rankhour_comparison_simple_build(traces[:2])
    plot_rankhour_comparison_simple_build(traces[:3])
    plot_rankhour_comparison_simple_build(traces[:4])


if __name__ == "__main__":
    #  global trace_dir_fmt
    trace_dir_fmt = "/mnt/ltio/parthenon-topo/{}"
    #  plot_init()
    plot_init_big()
    run()
