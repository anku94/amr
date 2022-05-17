import glob
import IPython
import multiprocessing
import numpy as np
import pandas as pd
import sys
import traceback

prev_output_ts = 0

def find_func(df, func_name):
    ts_begin = df[(df['func'] == func_name) & (df['enter_or_exit'] == 0)]['timestep']
    ts_end = df[(df['func'] == func_name) & (df['enter_or_exit'] == 1)]['timestep']
    assert(ts_begin.size == ts_end.size)
    all_invocations =  list(zip(ts_begin.array, ts_end.array))

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
        'AR3': ['Task_EstimateTimestep', 'LoadBalancingAndAdaptiveMeshRefinement'],
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
    df = pd.read_csv(in_path, usecols=range(4), lineterminator='\n', low_memory=False)
    df['group'] = np.where((df['func'] == 'MakeOutputs') & (df['enter_or_exit'] == '1'), 1, 0)
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
    run(host_idx)

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
    pdf = pdf.groupby(['ts','evtname'], as_index=False).agg({
        'evtval': ['count', 'mean', 'min', 'max', percentile(50), percentile(75), percentile(99)]
    })
    pdf.columns = ['_'.join(col).strip('_') for col in pdf.columns.values]
    pdf.to_csv(pdf_out, index=None)

def run_aggregate():
    phase_dir = '/mnt/lustre/parthenon-topo/profile3.min/phases'
    phase_df = read_phases(phase_dir)
    print(phase_df)
    analysis_df_path = '{}/aggregate.csv'.format(phase_dir)
    analyze_phase_df(phase_df, analysis_df_path)

if __name__ == '__main__':
    #  run_construct()
    run_aggregate()
