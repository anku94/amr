import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


class LabelAnalyzer:
    def __init__(self, dpath, label, stat_path):
        self.dpath_ = dpath
        self.label_ = label
        self.csv_path_ = '{}/by-label/{}.csv'.format(dpath, label)
        self.stat_path_ = stat_path
        self.stat_file_path_ = '{}/{}.stats.pickle'.format(self.stat_path_,
                                                           self.label_)

        try:
            os.mkdir(stat_path)
        except Exception as e:
            # yolo
            pass

        self.all_sums_ = {}
        self.all_gpc_ = {}
        self.all_msim_ = {}

    # statistics of interest:
    # for each timestep, total traffic
    # for each timestep, for each rank, unique nbrs, and median/std of this metric
    def compute_group_sum(self, ts, group):
        a = group['msg_sz_count']
        b = group['msg_sz_mean']
        ts_sum = np.dot(a, b)

        self.all_sums_[ts] = ts_sum

    def compute_group_peercnt(self, ts, group):
        stat_group = group.groupby('rank').agg({
            'peer': 'nunique'
        })

        mean, std = (stat_group['peer'].mean(), stat_group['peer'].std())
        self.all_gpc_[ts] = mean, std

    @staticmethod
    def get_comm_matrix(nranks, group):
        matrix = np.zeros([nranks, nranks], dtype=np.int64)
        for index, row in group.iterrows():
            src = row['rank']
            dest = row['peer']
            msg_sz = row['msg_sz_count'] * row['msg_sz_mean']
            matrix[src][dest] += msg_sz

        return matrix

    def compute_matrix_sim(self, ts, a, b):
        assert (a.shape == b.shape)
        sim = np.minimum(a, b)
        dist = a - b

        sim_l1 = np.sum(np.abs(sim))
        dist_l1 = np.sum(np.abs(dist))

        self.all_msim_[ts] = (sim_l1, dist_l1)

    def persist_stats(self):
        all_stats = {
            'ts_sums': self.all_sums_,
            'ts_gpc': self.all_gpc_,
            'ts_msim': self.all_msim_
        }

        with open(self.stat_file_path_, 'wb+') as f:
            f.write(pickle.dumps(all_stats))

    def read_stats(self):
        data = None

        with open(self.stat_file_path_, 'rb') as f:
            data = pickle.loads(f.read())

        print(data)

    def analyze(self):
        df = pd.read_csv(self.csv_path_).groupby(['timestep'])
        all_ts = list(df.groups.keys())
        all_ts = sorted(all_ts)

        matrix_prev = None
        ts_prev = None

        for ts in all_ts:
            print('Processing {}'.format(ts))
            ts_g = df.get_group(ts)
            self.compute_group_sum(ts, ts_g)
            self.compute_group_peercnt(ts, ts_g)

            matrix_cur = self.get_comm_matrix(512, ts_g)

            if matrix_prev is not None and ts_prev == ts - 1:
                self.compute_matrix_sim(ts_prev, matrix_prev, matrix_cur)

            matrix_prev = matrix_cur
            ts_prev = ts

        self.persist_stats()


def run_label(label):
    dpath = "/mnt/lt20ad2/parthenon-topo/profile2.min"
    stats_path = dpath + "/stats"

    la = LabelAnalyzer(dpath, label, stats_path)
    # la.analyze()
    la.read_back("/Users/schwifty/repos/amr-data/stats")


def sparse_to_dense(d, max_len, def_val):
    return [
        d[i] if i in d else def_val for i in range(max_len)
    ]


def read_label(label, stat_dir):
    stat_fpath = '{}/{}.stats.pickle'.format(stat_dir, label)
    data = None

    with open(stat_fpath, 'rb') as f:
        data = pickle.loads(f.read())

    print(data.keys())
    return data


def get_key_data(all_data, key, def_val):
    max_ts = 0
    for label_data in all_data:
        key_data = label_data[key]
        max_ts_alt = max(key_data.keys())
        max_ts = max(max_ts, max_ts_alt)

    # uniform-ize
    all_uf_key_data = []
    for label_data in all_data:
        key_data = label_data[key]
        uf_key_data = sparse_to_dense(key_data, max_ts, def_val)
        all_uf_key_data.append(uf_key_data)

    return all_uf_key_data


def plot_ts_sums(labels, ts_data, plot_dir):
    fig, ax = plt.subplots(1, 1)
    data_x = range(len(ts_data[0]))

    for cur_label, cur_data in zip(labels, ts_data):
        print(cur_label)
        ax.plot(data_x, cur_data, label=cur_label)

    ax.set_title('Phasewise Communication For A 512-Rank Blast Wave Run')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Total Communication Across All Ranks')
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f} KB'.format(x / 1e6))
    ax.legend()
    # fig.show()
    fig.savefig('{}/phasewise_comm.pdf'.format(plot_dir), dpi=300)
    ax.set_xlim([4950, 5050])
    fig.savefig('{}/phasewise_comm_zoomed.pdf'.format(plot_dir), dpi=300)


def plot_gpc(labels, gpc_data, plot_dir):
    fig, ax = plt.subplots(1, 1)
    data_x = range(len(gpc_data[0]))

    cm = plt.cm.get_cmap('Paired')

    idx = 0
    for cur_label, cur_data in zip(labels, gpc_data):
        data_mean, data_std = zip(*cur_data)
        data_mean = np.array(data_mean)
        data_std = np.array(data_std)
        ax.plot(data_x, data_mean, label=cur_label, color=cm(idx*2+1))
        ax.fill_between(data_x, data_mean - data_std, data_mean + data_std,
                        color=cm(idx*2+1), alpha=0.3)
        idx += 1

    ax.legend()
    ax.set_title('Peer-Counts (Mean/Std) for a 512-Rank AMR Run')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Peer Count')
    # fig.show()
    fig.savefig('{}/peer_count.pdf'.format(plot_dir), dpi=300)


def plot_all_labels(labels, stat_dir, plot_dir):
    all_label_data = []
    for label in labels:
        label_data = read_label(label, stat_dir)
        all_label_data.append(label_data)

    # ts_data = get_key_data(all_label_data, 'ts_sums', 0)
    # plot_ts_sums(labels, ts_data, plot_dir)

    gpc_data = get_key_data(all_label_data, 'ts_gpc', (0, 0))
    plot_gpc(labels, gpc_data, plot_dir)


def run():
    all_labels = ['FluxExchange', 'LoadBalancing', 'BoundaryComm']

    #  run_label(all_labels[0])
    #  run_label(all_labels[1])
    #  run_label(all_labels[2])

    stat_dir = '/Users/schwifty/repos/amr-data/stats'
    plot_dir = 'figures_bigrun'
    plot_all_labels(all_labels, stat_dir, plot_dir)


if __name__ == '__main__':
    run()
