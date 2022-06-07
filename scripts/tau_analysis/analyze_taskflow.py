import glob
import multiprocessing
import numpy as np
import pandas as pd
import subprocess
import sys
import traceback
from typing import Tuple

prev_output_ts = 0

def find_func_remove_mismatch(ts_begin, ts_end):
    all_inv = []

    for ts in ts_begin:
        all_inv.append((ts, 0))

    for ts in ts_end:
        all_inv.append((ts, 1))

    all_inv = sorted(all_inv)

    all_begin_clean = []
    all_end_clean = []

    prev_type = 1
    prev_ts = None

    for ts, ts_type in all_inv:
        assert(prev_type in [0, 1])
        assert(ts_type in [0, 1])

        if ts_type == 0:
            prev_type = 0
            prev_ts = ts

        if prev_type == 0 and ts_type == 1:
            all_begin_clean.append(prev_ts)
            all_end_clean.append(ts)
            prev_type = 1
            prev_ts = None

    return all_begin_clean, all_end_clean

def find_func(df, func_name):
    #  print('finding {}'.format(func_name))

    ts_begin = df[(df['func'] == func_name) & (df['enter_or_exit'] == 0)][
        'timestep']
    ts_end = df[(df['func'] == func_name) & (df['enter_or_exit'] == 1)][
        'timestep']
    #  print(ts_begin)
    #  print(ts_end)

    all_invocations = []

    try:
        assert (ts_begin.size == ts_end.size)
        all_invocations = list(zip(ts_begin.array, ts_end.array))
    except AssertionError as e:
        print(func_name, ts_begin.size, ts_end.size)
        print(traceback.format_exc())
        ts_begin, ts_end = find_func_remove_mismatch(ts_begin.array, ts_end.array)
        all_invocations = list(zip(ts_begin, ts_end))

    #  print(func_name)
    #  print(all_invocations)

    return all_invocations


def add_to_ts_beg(all_phases, phase_ts, phase_label):
    for pbeg, _ in phase_ts:
        all_phases.append((pbeg, phase_label + '.BEGIN'))


def add_to_ts_end(all_phases, phase_ts, phase_label):
    for _, pend in phase_ts:
        all_phases.append((pend, phase_label + '.END'))


def filter_phases(phases):
    phases = sorted(phases)

    prev_begin = False
    prev_name = None
    prev_ts = None
    
    lb_active = False
    lb_start_ts = None

    filtered_phases = []

    for phase_ts, phase_name in phases:
        cur_begin = False
        cur_name = phase_name.split('.')[0]

        if phase_name.endswith('.BEGIN'):
            cur_begin = True
        elif phase_name.endswith('.END'):
            cur_begin = False

        if lb_active and not cur_name.startswith('AR3'):
            continue
        elif cur_name == 'SR' and cur_begin == True:
            if prev_name == 'SR' and prev_begin == True:
                continue

        if cur_begin == False:
            if lb_active and cur_name == 'AR3':
                #  filtered_phases.append((lb_start_ts, 'AR3.BEGIN'))
                filtered_phases.append((phase_ts, 'AR3.END'))

                lb_active = False
                lb_start_ts = None
                continue

            if prev_begin == True:
                filtered_phases.append((prev_ts, prev_name + '.BEGIN'))
                filtered_phases.append((phase_ts, cur_name + '.END'))

            prev_begin = False
            prev_name = None
            prev_ts = None
        else:
            if prev_begin == True and prev_name == cur_name:
                continue

            if prev_begin == True:
                filtered_phases.append((prev_ts, prev_name + '.BEGIN'))

            prev_begin = True
            prev_name = cur_name
            prev_ts = phase_ts

            if cur_name == 'AR3':
                lb_active = True
                lb_start_ts = phase_ts

    return filtered_phases


def filter_phases_insert_missing(phases):
    new_phases = []

    lb_active = False

    prev_begin = False
    prev_phase = None
    prev_ts = None

    for phase_ts, phase_name in phases:
        cur_begin = False
        cur_name = phase_name.split('.')[0]

        if phase_name.endswith('.BEGIN'):
            cur_begin = True
        elif phase_name.endswith('.END'):
            cur_begin = False
        else:
            assert(False)

        if cur_name == 'AR3':
            new_phases.append((phase_ts, phase_name))
            continue

        if cur_begin == True:
            if prev_begin and prev_phase == 'AR1':
                new_phases.append((prev_ts, 'AR1.BEGIN'))
                new_phases.append((phase_ts - 1, 'AR1.END'))

            prev_begin = True
            prev_phase = cur_name
            prev_ts = phase_ts
        elif cur_begin == False:
            if prev_begin == True and prev_phase == cur_name:
                new_phases.append((prev_ts, cur_name + '.BEGIN'))
                new_phases.append((phase_ts, cur_name + '.END'))
            else:
                # prev_end and cur_begin both missing?
                # OR just cur_begin missing. Both not handled yet
                pass

            prev_begin = False
            prev_phase = None
            prev_ts = False
        else:
            assert(False)


    return new_phases


def validate_phases(phases):
    phases = sorted(phases)

    cur_stack = []

    for phase_ts, phase_name in phases:
        cur_begin = False
        cur_name = phase_name.split('.')[0]

        if phase_name.endswith('.BEGIN'):
            cur_begin = True
        elif phase_name.endswith('.END'):
            cur_begin = False
        else:
            assert(False)

        if cur_begin:
            cur_stack.append(cur_name)
        else:
            assert(len(cur_stack) > 0)
            cur_open_name = cur_stack.pop()
            assert(cur_open_name == cur_name)

    assert(len(cur_stack) == 0)


        
def classify_phases(df):
    phases = []

    phase_boundaries = {
        'AR1': ['Task_StartReceiving', 'FluxDivergenceBlock'],
        'AR2': ['Task_ClearBoundary', 'Task_FillDerived'],
        'AR3': ['Task_EstimateTimestep',
                'LoadBalancingAndAdaptiveMeshRefinement'],
        'SR': [None, 'Task_SetBoundaries_MeshData']
    }

    phase_boundaries = {
        'AR1': ['Reconstruct', 'Task_ReceiveFluxCorrection'],
        'AR2': ['Task_ClearBoundary', 'Task_FillDerived'],
        'AR3': ['LoadBalancingAndAdaptiveMeshRefinement',
                'LoadBalancingAndAdaptiveMeshRefinement'],
        'AR3_UMBT': ['UpdateMeshBlockTree',
                'UpdateMeshBlockTree'],
        'SR': ['Task_SendBoundaryBuffers_MeshData', 'Task_SetBoundaries_MeshData']
    }

    for phase, bounds in phase_boundaries.items():
        if bounds[0] is not None:
            ret = find_func(df, bounds[0])
            add_to_ts_beg(phases, ret, phase)
        if bounds[1] is not None:
            ret = find_func(df, bounds[1])
            add_to_ts_end(phases, ret, phase)

    
    phases = sorted(phases)

    def print_phases(phases, sep1, sep2):
        print(sep1 * 20)
        for phase in phases:
            print(phase)
        print(sep2 * 20)


    #  print_phases(phases, '=', '-')
    phases = filter_phases(phases)
    #  print_phases(phases, '-', '-')
    phases = filter_phases_insert_missing(phases)
    #  print_phases(phases, '-', '=')

    validation_passed = True

    try:
        validate_phases(phases)
    except AssertionError as e:
        print(traceback.format_exc())
        validation_passed = False
        print('VALIDATION FAILED!!!!')

    return phases, validation_passed


def aggregate_phases(df, phases):
    phase_total = {}
    cur_phase = None
    cur_phase_begin = -1

    def add_phase(phase, phase_time):
        if phase in phase_total:
            phase_total[phase] += phase_time
        else:
            phase_total[phase] = phase_time

    active_phases = {}

    for phase_ts, phase in phases:
        phase_name = phase.split('.')[0]

        if phase.endswith('.BEGIN'):
            active_phases[phase_name] = phase_ts
        elif phase.endswith('.END'):
            if phase_name in active_phases:
                phase_time = phase_ts - active_phases[phase_name]
                del active_phases[phase_name]
                add_phase(phase_name, phase_time)

    #  print(phase_total)
    total_phasewise = 0
    for key in ['AR1', 'AR2', 'AR3', 'AR3_UMBT', 'SR']:
        if key in phase_total:
            total_phasewise += phase_total[key]

    total_ts = df['timestep'].max() - df['timestep'].min()

    #  print('Phases: {}, Total: {}, Accounted: {:.0f}%'.format(total_phasewise, total_ts, total_phasewise * 100.0 / total_ts))

    return phase_total, total_phasewise, total_ts


def log_event(f, rank, ts, evt_name, evt_val):
    f.write('{:d},{:d},{},{:d}\n'.format(rank, ts, evt_name, evt_val))


def process_df_for_ts(rank, ts, df_ts, f):
    phases, validation_passed = classify_phases(df_ts)

    if validation_passed == False:
        print('Validation Failed: Rank {}, TS {}'.format(rank, ts))
        print(df_ts.to_string())
        sys.exit(-1)

    #  if (validation_passed == False):
        #  print(df_ts.to_string())
    #  print(df_ts.to_string())

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


def classify_trace(rank, in_path, out_path):
    df = pd.read_csv(in_path, usecols=range(4), lineterminator='\n',
                     low_memory=False)
    df = df.dropna().astype({
        'rank': 'int32',
        'timestep': 'int64',
        'func': str,
        'enter_or_exit': str
    })

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
            #  print(df_ts.to_string())
            #  df_ts = all_dfs.get_group(30)
            try:
                df_ts = df_ts[df_ts['enter_or_exit'].isin(['0', '1'])]
                df_ts = df_ts.dropna().astype({
                    'rank': 'int32',
                    'timestep': 'int64',
                    'func': str,
                    'enter_or_exit': int
                })
                #  df_ts = df_ts[df_ts['func'] != 'Task_ReceiveFluxCorrection']
                #  df_ts = df_ts[df_ts['func'] != 'ReceiveFluxCorrection_x1']
                process_df_for_ts(rank, ts, df_ts, f)
            except Exception as e:
                print(rank, ts)
                #  print(df_ts.to_string())
                print(e)
                print(traceback.format_exc())
                sys.exit(-1)
            #  if ts == 1: break
            #  if ts > 1000: break


def classify_trace_parworker(args):
    trace_in = args['in']
    trace_out = args['out']
    rank = args['rank']
    print('Parsing {} into {}...'.format(trace_in, trace_out))
    classify_trace(rank, trace_in, trace_out)


def classify_parallel(dpath, all_ranks):
    print('Processing ranks {} to {}'.format(all_ranks[0], all_ranks[-1]))

    all_args = []
    for rank in all_ranks:
        cur_arg = {}
        cur_arg['rank'] = rank
        cur_arg['in'] = '{}/trace/funcs.{}.csv'.format(dpath, rank)
        cur_arg['out'] = '{}/phases/phases.{}.csv'.format(dpath, rank)
        all_args.append(cur_arg)

    with multiprocessing.Pool(16) as pool:
        pool.map(classify_trace_parworker, all_args)


def run_classify_serial():
    rank = 147
    basedir = '/mnt/ltio/parthenon-topo/profile6.wtau'
    in_path = '{}/trace/funcs.{}.csv'.format(basedir, rank)
    in_path = '{}/trace/tmp2.csv'.format(basedir)
    out_path = '{}/phases/phases.{}.csv'.format(basedir, rank)
    out_path = '{}/tmp.{}.csv'.format(basedir, rank)
    classify_trace(rank, in_path, out_path)


def run_classify_parallel():
    host_idx, num_hosts = get_exp_stats()
    print ('Node {} of {}'.format(host_idx, num_hosts))
    if host_idx > 31:
        print('Nothing to do')
        return
    #  host_idx = int(sys.argv[1][1:])

    dpath = '/mnt/lustre/parthenon-topo/profile3.min'
    dpath = '/mnt/lustre/parthenon-topo/profile4/trace'
    dpath = '/mnt/ltio/parthenon-topo/profile6.wtau'
    ranks_per_node = 16
    rbeg = ranks_per_node * host_idx
    rend = rbeg + ranks_per_node

    ranks_to_process = list(range(rbeg, rend))

    print('Processing: {}'.format(', '.join([str(i) for i in ranks_to_process])))

    classify_parallel(dpath, ranks_to_process)


def read_phases(pdir):
    all_phase_csvs = glob.glob(pdir + '/phases/phases.*.csv')
    print(len(all_phase_csvs))

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
    #  pdf = pdf.groupby(['ts', 'evtname'], as_index=False).agg({
        #  'evtval': ['count', 'mean', 'min', 'max', percentile(50),
                   #  percentile(75), percentile(99)]
    #  })
    #  pdf = pdf[pdf['ts'] < 10]

    pdf = (
        pdf.sort_values(['ts', 'evtname', 'rank'])
        .groupby(['ts', 'evtname'], as_index=False)
        .agg({
        'evtval': list
    }))
    print(pdf)
    #  pdf.columns = ['_'.join(col).strip('_') for col in pdf.columns.values]
    pdf.to_csv(pdf_out, index=None)


def run_parse_log(dpath: str):
    f = open('{}/run/log.txt'.format(dpath)).read().split('\n')
    data = [line for line in f if 'cycle' in line]

    keys = [i.split('=')[0] for i in data[0].split(' ')]
    vals = [[float(i.split('=')[1]) for i in data[k].split(' ')]
            for k in range(len(data))]
    print(keys)
    df = pd.DataFrame.from_records(vals, columns=keys)
    print(df)
    df.to_csv('{}/logstats.csv'.format(dpath), index=None)


def run_aggregate():
    phase_dir = '/mnt/ltio/parthenon-topo/profile6.wtau'
    #  run_parse_log(phase_dir)
    #  return

    phase_df = read_phases(phase_dir)
    print(phase_df)
    analysis_df_path = '{}/aggregate.csv'.format(phase_dir)
    analyze_phase_df(phase_df, analysis_df_path)


def get_exp_stats() -> Tuple[int, int]:
    result = subprocess.run(['/share/testbed/bin/emulab-listall'], stdout=subprocess.PIPE)
    hoststr = str(result.stdout.decode('ascii'))
    hoststr = hoststr.strip().split(',')
    num_hosts = len(hoststr)
    our_id = open('/var/emulab/boot/nickname', 'r').read().split('.')[0][1:]
    our_id = int(our_id)

    #  print(our_id, num_hosts)

    return (our_id, num_hosts)


if __name__ == '__main__':
    #  run_classify_serial()
    #  run_classify_parallel()
    run_aggregate()
