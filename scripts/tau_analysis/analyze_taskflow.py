import multiprocessing
import numpy as np
import pandas as pd
import IPython

def find_func(df, func_name):
    ts_begin = df[(df['func'] == func_name) & (df['enter_or_exit'] == '0')]['timestep']
    ts_end = df[(df['func'] == func_name) & (df['enter_or_exit'] == '1')]['timestep']
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

    print(phase_total)
    total_phasewise = 0
    for key in ['AR1', 'AR2', 'AR3', 'SR']:
        if key in phase_total:
            total_phasewise += phase_total[key]

    total_ts = df['timestep'].max() - df['timestep'].min()

    print('Phases: {}, Total: {}, Accounted: {:.0f}%'.format(total_phasewise, total_ts, total_phasewise * 100.0 / total_ts))

    return phase_total, total_phasewise, total_ts

def log_event(f, ts, evt_name, evt_val):
    f.write('{:d},{},{:d}\n'.format(ts, evt_name, evt_val))

def write_out(f, ts, phase_total, total_phasewise, total_ts):
    for phase, phase_time in phase_total.items():
        log_event(f, ts, phase, phase_time)
    log_event(f, ts, 'PHASE_TOTAL', total_phasewise)
    log_event(f, ts, 'ACTUAL_TOTAL', total_ts)

def process_df_for_ts(ts, df_ts, f):
    phases = classify_phases(df_ts)
    phase_total, total_phases, total_ts = aggregate_phases(df_ts, phases)
    write_out(f, ts, phase_total, total_phases, total_ts)

def analyze_trace(in_path, out_path):
    df = pd.read_csv(in_path, usecols=range(4), lineterminator='\n')
    df['group'] = np.where((df['func'] == 'MakeOutputs') & (df['enter_or_exit'] == '1'), 1, 0)
    df['group'] = df['group'].shift(1).fillna(0).astype(int)
    df['group'] = df['group'].cumsum()
    all_dfs = df.groupby('group', as_index=False)

    with open(out_path, 'w+') as f:
        header = 'ts,evtname,evtval\n'
        f.write(header)
        #  for ts, df_ts in all_dfs:
            #  print(ts)
            #  process_df_for_ts(ts, df_ts, f)
        cur_ts = 101
        df = all_dfs.get_group(cur_ts)
        process_df_for_ts(cur_ts, df, f)

def analyze_worker(args):
    trace_in = args['in']
    trace_out = args['out']
    print('Parsing {} into {}...'.format(trace_in, trace_out))

def run_parallel(dpath, num_ranks):
    all_args = []
    for rank in range(num_ranks):
        cur_arg = {}
        cur_arg['in'] = '{}/funcs.{}.csv'.format(dpath, rank)
        cur_arg['out'] = '{}/phases.{}.csv'.format(dpath, rank)
        all_args.append(cur_arg)

    with multiprocessing.Pool(4) as pool:
        pool.map(analyze_worker, all_args)


def run():
    in_path = '/mnt/lustre/parthenon-topo/tmp/tmp.csv'
    out_path = '/mnt/lustre/parthenon-topo/tmp/tmp.phase.csv'
    analyze_trace(in_path, out_path)

    #  dpath = '/root/profile3'
    #  run_parallel(dpath, 16)

if __name__ == '__main__':
    run()
