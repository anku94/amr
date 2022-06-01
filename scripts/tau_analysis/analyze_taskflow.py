import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import multiprocessing
import numpy as np
import pandas as pd
import sys
import traceback

prev_output_ts = 0


def find_func(df, func_name):
    ts_begin = df[(df['func'] == func_name) & (df['enter_or_exit'] == 0)][
        'timestep']
    ts_end = df[(df['func'] == func_name) & (df['enter_or_exit'] == 1)][
        'timestep']
    assert (ts_begin.size == ts_end.size)
    all_invocations = list(zip(ts_begin.array, ts_end.array))

    #  print(func_name)
    #  print(all_invocations)

    return all_invocations


def add_to_ts_beg(all_phases, phase_ts, phase_label):
    for pbeg, _ in phase_ts:
        all_phases.append((pbeg, phase_label + '.BEGIN'))


def add_to_ts_end(all_phases, phase_ts, phase_label):
    for _, pend in phase_ts:
        all_phases.append((pend, phase_label + '.END'))


def classify_phases(df):
    phases = []

    phase_boundaries = {
        'AR1': ['Task_StartReceiving', 'FluxDivergenceBlock'],
        'AR2': ['Task_ClearBoundary', 'Task_FillDerived'],
        'AR3': ['Task_EstimateTimestep',
                'LoadBalancingAndAdaptiveMeshRefinement'],
        'SR': [None, 'Task_SetBoundaries_MeshData']
    }

    for phase, bounds in phase_boundaries.items():
        if bounds[0] is not None:
            ret = find_func(df, bounds[0])
            add_to_ts_beg(phases, ret, phase)
        if bounds[1] is not None:
            ret = find_func(df, bounds[1])
            add_to_ts_end(phases, ret, phase)

    phases = sorted(phases)
    #  for phase in phases:
    #  print(phase)

    return phases


def aggregate_phases(df, phases):
    phase_total = {}
    cur_phase = None
    cur_phase_begin = -1

    def add_phase(phase, phase_time):
        if phase in phase_total:
            phase_total[phase] += phase_time
        else:
            phase_total[phase] = phase_time

    for phase_ts, phase in phases:
        if phase.endswith('.BEGIN'):
            if cur_phase is not None:
                cur_phase_time = phase_ts - cur_phase_begin
                add_phase(cur_phase, cur_phase_time)
            cur_phase = phase[:-6]
            cur_phase_begin = phase_ts
        elif phase.endswith('.END'):
            if cur_phase is not None:
                cur_phase_time = phase_ts - cur_phase_begin
                add_phase(cur_phase, cur_phase_time)
            cur_phase = phase[:-4]
            cur_phase_begin = phase_ts

    #  print(phase_total)
    total_phasewise = 0
    for key in ['AR1', 'AR2', 'AR3', 'SR']:
        if key in phase_total:
            total_phasewise += phase_total[key]

    total_ts = df['timestep'].max() - df['timestep'].min()

    #  print('Phases: {}, Total: {}, Accounted: {:.0f}%'.format(total_phasewise, total_ts, total_phasewise * 100.0 / total_ts))

    return phase_total, total_phasewise, total_ts


def log_event(f, rank, ts, evt_name, evt_val):
    f.write('{:d},{:d},{},{:d}\n'.format(rank, ts, evt_name, evt_val))


def process_df_for_ts(rank, ts, df_ts, f):
    phases = classify_phases(df_ts)
    phase_total, total_phasewise, total_ts = aggregate_phases(df_ts, phases)

    global prev_output_ts

    output_call = find_func(df_ts, 'MakeOutputs')
    cur_output_ts = output_call[0][1]

    for phase, phase_time in phase_total.items():
        log_event(f, rank, ts, phase, phase_time)

    log_event(f, rank, ts, 'TIME_CLASSIFIEDPHASES', total_phasewise)
    log_event(f, rank, ts, 'TIME_FROMCURBEGIN', total_ts)
    log_event(f, rank, ts, 'TIME_FROMPREVEND', cur_output_ts - prev_output_ts)

    prev_output_ts = cur_output_ts


def analyze_trace(rank, in_path, out_path):
    df = pd.read_csv(in_path, usecols=range(4), lineterminator='\n',
                     low_memory=False)
    df['group'] = np.where(
        (df['func'] == 'MakeOutputs') & (df['enter_or_exit'] == '1'), 1, 0)
    df['group'] = df['group'].shift(1).fillna(0).astype(int)
    df['group'] = df['group'].cumsum()
    all_dfs = df.groupby('group', as_index=False)

    with open(out_path, 'w+') as f:
        header = 'rank,ts,evtname,evtval\n'
        f.write(header)
        for ts, df_ts in all_dfs:
            #  print(ts)
            try:
                df_ts = df_ts[df_ts['enter_or_exit'].isin(['0', '1'])]
                df_ts = df_ts.dropna().astype({
                    'rank': 'int32',
                    'timestep': 'int64',
                    'func': str,
                    'enter_or_exit': int
                })
                process_df_for_ts(rank, ts, df_ts, f)
            except Exception as e:
                print(rank, ts)
                print(df_ts.to_string())
                print(e)
                print(traceback.format_exc())
        #  cur_ts = 101
        #  df = all_dfs.get_group(cur_ts)
        #  process_df_for_ts(cur_ts, df, f)


def analyze_worker(args):
    trace_in = args['in']
    trace_out = args['out']
    rank = args['rank']
    print('Parsing {} into {}...'.format(trace_in, trace_out))
    analyze_trace(rank, trace_in, trace_out)


def run_parallel(dpath, all_ranks):
    print('Processing ranks {} to {}'.format(all_ranks[0], all_ranks[-1]))

    all_args = []
    for rank in all_ranks:
        cur_arg = {}
        cur_arg['rank'] = rank
        cur_arg['in'] = '{}/funcs.{}.csv'.format(dpath, rank)
        cur_arg['out'] = '{}/phases/phases.{}.csv'.format(dpath, rank)
        all_args.append(cur_arg)

    with multiprocessing.Pool(16) as pool:
        pool.map(analyze_worker, all_args)


def construct_phases(hidx):
    #  rank = 59
    #  in_path = '/mnt/lustre/parthenon-topo/profile3.min/funcs.{}.csv'.format(59)
    #  out_path = '/mnt/lustre/parthenon-topo/tmp/tmp.phase.csv'
    #  analyze_trace(rank, in_path, out_path)

    dpath = '/mnt/lustre/parthenon-topo/profile3.min'
    ranks_per_node = 16
    rbeg = ranks_per_node * hidx
    rend = rbeg + ranks_per_node
    ranks_to_process = list(range(rbeg, rend))
    run_parallel(dpath, ranks_to_process)


def run_construct():
    host_idx = int(sys.argv[1][1:])
    construct_phases(host_idx)


def read_phases(pdir):
    all_phase_csvs = glob.glob(pdir + '/phases*.csv')

    #  all_phase_csvs = all_phase_csvs[8:12]
    def read_phase_df(x):
        df = pd.read_csv(x).astype({
            'rank': 'int32',
            'ts': 'int32',
            'evtname': str,
            'evtval': 'int64'
        })
        return df

    all_dfs = map(read_phase_df, all_phase_csvs)
    aggr_df = pd.concat(all_dfs).dropna()
    return aggr_df


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def analyze_phase_df(pdf, pdf_out):
    pdf = pdf.groupby(['ts', 'evtname'], as_index=False).agg({
        'evtval': ['count', 'mean', 'min', 'max', percentile(50),
                   percentile(75), percentile(99)]
    })
    pdf.columns = ['_'.join(col).strip('_') for col in pdf.columns.values]
    pdf.to_csv(pdf_out, index=None)


def run_aggregate():
    phase_dir = '/mnt/lustre/parthenon-topo/profile3.min/phases'
    phase_df = read_phases(phase_dir)
    print(phase_df)
    analysis_df_path = '{}/aggregate.csv'.format(phase_dir)
    analyze_phase_df(phase_df, analysis_df_path)


def plot_init():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_neighbors(df, plot_dir):
    fig, ax = plt.subplots(1, 1)
    print(df.describe())
    print(df.columns)

    df = df.groupby('ts', as_index=False).agg({
        'evtval_count': ['mean']
    })

    data_x = df['ts']
    data_y = df['evtval_count']['mean']

    ax.plot(data_x, data_y)
    ax.set_title('Datapoints Salvaged (Out of 512) For Each AMR TS')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number Of Datapoints (= Ranks) Parseable')
    # fig.show()
    fig.savefig('{}/taskflow_nbrcnt.pdf'.format(plot_dir), dpi=300)


def get_data(df, evt, col):
    df = df[df['evtname'] == evt]
    data_x = df['ts']
    data_y = df[col]
    return data_x, data_y


def plot_event(event_name, df, plot_dir, plot_tail=False, save=False):
    fig, ax = plt.subplots(1, 1)
    cm = plt.cm.get_cmap('tab20c')

    dx, dy = get_data(df, event_name, 'evtval_mean')
    ax.plot(dx, dy, color=cm(0), label='Mean ({})'.format(event_name))

    dx, dy = get_data(df, event_name, 'evtval_percentile_50')
    ax.plot(dx, dy, '--', color=cm(4),
            label='50th %-ile ({})'.format(event_name))

    dx, dy = get_data(df, event_name, 'evtval_percentile_75')
    ax.plot(dx, dy, '--', color=cm(8),
            label='75th %-ile ({})'.format(event_name))

    if plot_tail:
        dx, dy = get_data(df, event_name, 'evtval_percentile_99')
        ax.plot(dx, dy, '--', color=cm(12),
                label='99th %-ile ({})'.format(event_name))

    ax.set_title('Statistics for Event {}'.format(event_name))
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Time (s)')

    ax.legend()
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.1f}s'.format(x / 1e6))

    ax.set_xlim([4000, 6000])

    event_key = event_name.lower()
    event_key = '{}_zoomed'.format(event_name.lower())

    plot_fname = None
    if plot_tail:
        plot_fname = 'taskflow_{}_w99.pdf'.format(event_key)
    else:
        plot_fname = 'taskflow_{}_wo99.pdf'.format(event_key)

    fig.tight_layout()
    if save:
        fig.savefig('{}/{}'.format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()


def plot_all_events(df, plot_dir):
    for event in (['AR1', 'AR2', 'AR3', 'SR']):
        plot_event(event, df, plot_dir, plot_tail=False, save=True)
        plot_event(event, df, plot_dir, plot_tail=True, save=True)


def plot_amr_log(log_df, plot_dir, save=False):
    print(log_df)

    fig, ax = plt.subplots(1, 1)

    key_y = 'wtime_step_other'
    label_y = 'Walltime (Non-AMR/LB)'
    data_x = log_df['cycle']
    data_y = log_df[key_y]
    ax.plot(data_x, data_y, label=label_y)

    key_y = 'wtime_step_amr'
    label_y = 'Walltime (AMR/LB)'
    data_x = log_df['cycle']
    data_y = log_df[key_y]
    ax.plot(data_x, data_y, label=label_y)

    ax.set_title('Wall Time for AMR Run (512 Timesteps)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Walltime (seconds)')

    ax.legend()

    plot_fname = 'amr_steptimes.pdf'

    # ax.set_xlim([3750, 4250])
    # plot_fname = 'amr_steptimes_zoomed.pdf'

    fig.tight_layout()

    if save:
        fig.savefig('{}/{}'.format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()


def calc_amr_log_stats(log_df):
    def calc_key_stats(key):
        print('Analyzing {}'.format(key))
        data_y = log_df[key]
        med_val = np.median(data_y)
        sum_val = data_y.sum()
        print('Median: {:.2f}, Sum: {:.2f}'.format(med_val, sum_val))

        first_half = sum([i for i in data_y if i < med_val])
        second_half = sum([i for i in data_y if i > med_val])

        print('Sums: {:.1f}/{:.1f} (First 50%, Last 50%)'.format(first_half,
                                                                 second_half))

    data_y = log_df['wtime_step_other']
    calc_key_stats('wtime_step_other')
    calc_key_stats('wtime_step_amr')


def plot_amr_log_distrib(log_df, plot_dir, save=False):
    fig, ax = plt.subplots(1, 1)

    data_y = log_df['wtime_step_other']
    plt.hist(data_y, bins=100, density=0, histtype='step', cumulative=True,
             label='Non-AMR/LB (Cumul.)')
    # plt.hist(data_y, bins=100, density=0, histtype='step', cumulative=False, label='Non-AMR/LB')
    data_y = log_df['wtime_step_amr']
    plt.hist(data_y, bins=100, density=0, histtype='step', cumulative=True,
             label='AMR/LB (Cumul)')
    # plt.hist(data_y, bins=100, density=0, histtype='step', cumulative=False, label='AMR/LB')

    ax.legend()

    noncum_profile = True
    zoomed_profile = False
    save = False

    if noncum_profile:
        ax.set_xlim([0, 3])
        ax.set_title('Wall Time for AMR Run (512 Timesteps)')
        ax.set_xlabel('Walltime (seconds)')
        ax.set_ylabel('Num Times')
        plot_fname = 'amr_steptimes_distrib_noncumul.pdf'
    else:
        ax.set_title('Wall Time for AMR Run (512 Timesteps)')
        ax.set_xlabel('Walltime (seconds)')
        ax.set_ylabel('Number Of Timesteps > X')

        ax.yaxis.set_major_formatter(
            lambda x, pos: max(round(30000 * (1 - x)), 0))

        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        if zoomed_profile:
            ax.set_ylim([0.99, 1.001])
            plot_fname = 'amr_steptimes_distrib.pdf'
            ax.yaxis.set_major_locator(MultipleLocator(0.002))
            ax.yaxis.set_minor_locator(MultipleLocator(0.001))
        else:
            plot_fname = 'amr_steptimes_distrib.pdf'
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.grid(visible=True, which='major', color='#999')
    plt.grid(visible=True, which='minor', color='#ddd')
    fig.tight_layout()

    if save:
        fig.savefig('{}/{}'.format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()


def plot_amr_comp(all_dfs, plot_dir, save=False):
    fig, ax = plt.subplots(1, 1)

    cm = plt.cm.get_cmap('Set1')

    for idx, df in enumerate(all_dfs):
        data_x = df['cycle']
        data_y1 = df['wtime_step_other']
        data_y2 = df['wtime_step_amr']

        label_1 = 'Run{}-Kernel'.format(idx)
        label_2 = 'Run{}-AMR'.format(idx)
        ax.plot(data_x, data_y1.cumsum(), label=label_1, color=cm(idx))
        ax.plot(data_x, data_y2.cumsum(), label=label_2, linestyle='--',
                color=cm(idx))

    ax.set_title('AMR Runs (512 Ranks) Phasewise Cumul. Times')
    ax.set_xlabel('Timestep')
    ax.set_xlabel('Total Time (seconds)')

    ax.legend()
    plt.grid(visible=True, which='major', color='#999')
    plt.grid(visible=True, which='minor', color='#ddd')
    fig.tight_layout()

    plot_fname = 'amr_steptimes_comp.pdf'
    plot_fname = 'amr_steptimes_comp_zoomed.pdf'
    ax.set_xlim([0000, 10000])

    # save = True

    if save:
        fig.savefig('{}/{}'.format(plot_dir, plot_fname), dpi=300)
    else:
        fig.show()
    pass


def run_plot_amr_comp():
    plot_dir = 'figures_bigrun'
    log_dirs = [
        '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log.times.csv',
        '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log2.csv',
        '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log3.csv',
        '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log4.csv'
    ]

    log_dfs = map(pd.read_csv, log_dirs)
    plot_amr_comp(log_dfs, plot_dir, save=False)


def plot_profile():
    pass


def run_profile():
    df_path = '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/profile.log.csv'
    df = pd.read_csv(df_path)
    df = df.astype({
        'rank': 'int32',
        'event': str,
        'timepct': float
    })
    events = df['event'].unique()

    fig, ax = plt.subplots(1, 1)

    for event in events:
        dfe = df[df['event'] == event]
        data_x = dfe['rank']
        data_y = dfe['timepct']
        print(data_x)
        print(data_y)
        ax.plot(dfe['rank'], dfe['timepct'], label=event)

    ax.set_title('Function-Wise Times (Pct Of Process Time)')
    ax.set_xlabel('Rank Index')
    ax.set_ylabel('Time Taken (%)')
    ax.legend()
    # fig.tight_layout()
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}%'.format(x))
    fig.show()
    plot_dir = 'figures_bigrun'
    plot_fname = 'amr_profile_phases.pdf'
    fig.savefig('{}/{}'.format(plot_dir, plot_fname), dpi=300)


def run_plot():
    # aggr_fpath = '/Users/schwifty/repos/amr-data/20220517-phase-analysis/aggregate.csv'
    # df = pd.read_csv(aggr_fpath)
    plot_init()
    plot_dir = 'figures_bigrun'
    # # plot_neighbors(df, plot_dir)
    # plot_all_events(df, plot_dir)

    # phoebus_log = '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log.times.csv'
    # phoebus_log2 = '/Users/schwifty/Repos/amr-data/20220524-phase-analysis/phoebus.log2.csv'
    # log_df = pd.read_csv(phoebus_log)
    # log_df2 = pd.read_csv(phoebus_log2)
    # plot_amr_log(log_df, plot_dir, save=True)
    # plot_amr_log_distrib(log_df, plot_dir, save=False)
    # calc_amr_log_stats(log_df)
    # run_plot_amr_comp()
    run_profile()


if __name__ == '__main__':
    #  run_construct()
    # run_aggregate()
    run_plot()
