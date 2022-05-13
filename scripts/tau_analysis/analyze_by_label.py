import os
import numpy as np
import pandas as pd
import pickle

class LabelAnalyzer:
    def __init__(self, dpath, label, stat_path):
        self.dpath_ = dpath
        self.label_ = label
        self.csv_path_ = '{}/by-label/{}.csv'.format(dpath, label)
        self.stat_path_ = stat_path
        self.stat_file_path_ = '{}/{}.stats.pickle'.format(self.stat_path_, self.label_)

        try:
            os.mkdir(stat_path)
        except FileExistsError as e:
            pass

        self.all_sums_ = {}
        self.all_gpc_ = {}
        self.all_msim_ = {}

    # statistics of interest:
    # for each timestep, total traffic
    # for each timestep, for each rank, unique nbrs, and median/std of this metric
    def ComputeGroupSum(self, ts, group):
        a = group['msg_sz_count']
        b = group['msg_sz_mean']
        ts_sum = np.dot(a, b)

        self.all_sums_[ts] = ts_sum

    def ComputeGroupPeerCnt(self, ts, group):
        stat_group = group.groupby('rank').agg({
            'peer': 'nunique'
        })

        mean, std = (stat_group['peer'].mean(), stat_group['peer'].std())
        self.all_gpc_[ts] = mean, std

    def GetCommMatrix(self, nranks, group):
        matrix = np.zeros([nranks, nranks], dtype=np.int64)
        for index, row in group.iterrows():
            src = row['rank']
            dest = row['peer']
            msg_sz = row['msg_sz_count'] * row['msg_sz_mean']
            matrix[src][dest] += msg_sz

        return matrix

    def ComputeMatrixSim(self, ts, a, b):
        assert (a.shape == b.shape)
        sim = np.minimum(a, b)
        dist = a - b

        sim_l1 = np.sum(np.abs(sim))
        dist_l1 = np.sum(np.abs(dist))

        self.all_msim_[ts] = (sim_l1, dist_l1)

    def PersistStats(self):
        all_stats = {
            'ts_sums': self.all_sums_,
            'ts_gpc': self.all_gpc_,
            'ts_msim': self.all_msim_
        }

        with open(self.stat_file_path_, 'wb+') as f:
            f.write(pickle.dumps(all_stats))

    def ReadStats(self):
        data = None

        with open(self.stat_file_path_, 'rb') as f:
            data = pickle.loads(f.read())

        print(data)

    def Analyze(self):
        df = pd.read_csv(self.csv_path_).groupby(['timestep'])
        all_ts = list(df.groups.keys())
        all_ts = sorted(all_ts)

        matrix_prev = None
        ts_prev = None

        for ts in all_ts:
            print('Processing {}'.format(ts))
            ts_g = df.get_group(ts)
            self.ComputeGroupSum(ts, ts_g)
            self.ComputeGroupPeerCnt(ts, ts_g)

            matrix_cur = self.GetCommMatrix(512, ts_g)

            if matrix_prev is not None and ts_prev == ts - 1:
                self.ComputeMatrixSim(ts_prev, matrix_prev, matrix_cur)

            matrix_prev = matrix_cur
            ts_prev = ts

        self.PersistStats()


def run_label(label):
    dpath = "/mnt/lt20ad2/parthenon-topo/profile2.min"
    stats_path = dpath + "/stats"

    la = LabelAnalyzer(dpath, label, stats_path)
    la.Analyze()
    la.ReadStats()


def run():
    #  run_label('FluxExchange')
    #  run_label('LoadBalancing')
    run_label('BoundaryComm')


if __name__ == '__main__':
    run()
