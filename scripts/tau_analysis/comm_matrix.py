import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt


class CommMatrix:
    def __init__(self, nranks):
        self.nranks = int(nranks)
        self.all_matrices = {}

    def EnsureFit(self, phase, timestep):
        shape_new = [int(timestep) + 1, self.nranks, self.nranks]

        if phase not in self.all_matrices:
            self.all_matrices[phase] = np.zeros(shape_new, dtype=np.int64)
        elif self.all_matrices[phase].shape[0] <= timestep:
            self.all_matrices[phase].resize(shape_new)

    def Add(self, phase, timestep, src, dest, msgsz):
        self.EnsureFit(phase, timestep)
        try:
            self.all_matrices[phase][timestep][src][dest] += msgsz
        except IndexError as e:
            print('error: ', phase, timestep, src, dest, msgsz)

    def Print(self):
        print(self.all_matrices)

    def Persist(self):
        persisted_state = [self.nranks, self.all_matrices]
        with open('.matrix', 'wb+') as f:
            f.write(pickle.dumps(persisted_state))

    def LoadPersisted(self):
        with open('.matrix', 'rb') as f:
            data_bytes = f.read()
            persisted_state = pickle.loads(data_bytes)
            self.nranks = persisted_state[0]
            self.all_matrices = persisted_state[1]

    def GetPhaseSums(self):
        ts_max = 0
        all_sums = {}

        for phase in self.all_matrices.keys():
            phase_matrix = self.all_matrices[phase]
            phase_sum = np.sum(phase_matrix, axis=(1, 2))

            ts_max = max(ts_max, phase_sum.shape[0])
            all_sums[phase] = phase_sum

        for phase in all_sums:
            all_sums[phase].resize(ts_max)

        return all_sums

    def PrintSummary(self):
        all_sums = self.GetPhaseSums()
        for phase in all_sums:
            print(phase, all_sums[phase])

    @staticmethod
    def GetMatrixSimilarityScores(a, b):
        assert (a.shape == b.shape)
        sim = np.minimum(a, b)
        dist = a - b

        sim_l1 = np.sum(np.abs(sim))
        dist_l1 = np.sum(np.abs(dist))
        sim_l2 = np.linalg.norm(sim)
        dist_l2 = np.linalg.norm(dist)

        print(sim_l1)
        print(dist_l1)
        print(sim_l2)
        print(dist_l2)

        return (sim_l1, dist_l1, sim_l2, dist_l2)

    def GetCommMatrixSimilarityScores(self, phase):
        phase_ts = self.all_matrices[phase]
        all_dist = []

        num_ts = phase_ts.shape[0]

        for i in range(0, num_ts - 1):
            mat_a = phase_ts[i]
            mat_b = phase_ts[i + 1]
            dist = CommMatrix.GetMatrixSimilarityScores(mat_a, mat_b)
            all_dist.append(dist)

        all_dist = list(zip(*all_dist))
        print(all_dist)
        return all_dist


def generate_matrix_rank(matrix, dpath, rank):
    rank_csv = "{}/log.{}.csv".format(dpath, rank)
    df = pd.read_csv(rank_csv, on_bad_lines='skip')
    df = df[df['send_or_recv'] == 0]
    df = df[['rank', 'peer', 'timestep', 'phase', 'msg_sz']]
    df = df.groupby(['rank', 'peer', 'timestep', 'phase'], as_index=False).agg(
        ['sum']).reset_index()
    df.columns = df.columns.to_flat_index().str.join('_')
    df.columns = df.columns.str.strip('_')

    df = df.astype({'rank': int, 'peer': int, 'timestep': int, 'phase': str,
                    'msg_sz_sum': np.int64})

    for index, row in df.iterrows():
        rank = row['rank']
        peer = row['peer']
        timestep = row['timestep']
        phase = row['phase']
        msgsz = row['msg_sz_sum']

        matrix.Add(phase, timestep, rank, peer, msgsz)


def generate_matrix(dpath):
    nranks = 512

    matrix = CommMatrix(nranks)

    for rank in range(nranks):
        print('Reading Rank {}'.format(rank))
        generate_matrix_rank(matrix, dpath, rank)

    matrix.PrintSummary()
    matrix.Persist()


def plot_matrix_phasewise_sums(all_sums, fig_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    for phase in all_sums:
        data_y = all_sums[phase]
        data_x = np.arange(len(data_y))
        ax.plot(data_x, data_y, label=phase)

    ax.legend()
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f} KB'.format(x / 1e6))
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Total Communication Per Timestep')
    ax.set_title('Phase-Wise Communication Breakdown of a 512-Rank Run')
    # fig.show()
    fig.tight_layout()
    fig.savefig(fig_dir + '/phasewise_comm.pdf', dpi=600)


def plot_matrix_sim_scores(scores, fig_dir) -> None:
    data_x = np.arange(len(scores[0]))

    fig, ax = plt.subplots(1, 1)

    # ax = axes[0]
    ax.plot(data_x, scores[0], label='Min-Matrix L1', linestyle='--')
    ax.plot(data_x, scores[1], label='Delta-Matrix L1')

    # ax = axes[1]
    # ax.plot(data_x, scores[2], label='Min-Matrix L2', linestyle='--')
    # ax.plot(data_x, scores[3], label='Delta-Matrix L2')
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f} KB'.format(x / 1e6))

    ax.legend()
    ax.set_title('Similarity Scores Between Adjacent Comm-Matrices (512-Rank AMR)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('L1 Norm of Comm-Matrix')
    # fig.show()
    fig.tight_layout()
    fig.savefig(fig_dir + '/comm_matrix_sim.pdf', dpi=600)


def analyze_matrix():
    nranks = 512
    fig_path = 'figures'

    matrix = CommMatrix(0)
    matrix.LoadPersisted()
    # matrix.PrintSummary()
    # matrix_sums = matrix.GetPhaseSums()
    # plot_matrix_phasewise_sums(matrix_sums, 'figures')
    scores = matrix.GetCommMatrixSimilarityScores('BoundaryComm')
    plot_matrix_sim_scores(scores, fig_path)


def run():
    dpath = "/mnt/lt20ad2/parthenon-topo/profile"
    #  generate_matrix(dpath)
    analyze_matrix()


if __name__ == '__main__':
    run()
