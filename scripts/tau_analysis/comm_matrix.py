import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt


class CommMatrix:
    def __init__(self, nranks):
        self.nranks = int(nranks)
        self.all_matrices = {}
        self.all_rank_msgcnt = {}
        self.all_rank_nbrcnt = {}

    def EnsureFit(self, phase, timestep):
        shape_new = [int(timestep) + 1, self.nranks, self.nranks]

        if phase not in self.all_matrices:
            self.all_matrices[phase] = np.zeros(shape_new, dtype=np.int64)
        elif self.all_matrices[phase].shape[0] <= timestep:
            self.all_matrices[phase].resize(shape_new)
            #  self.all_matrices[phase] = np.resize(self.all_matrices[phase], shape_new)

    def EnsureFit2D(self, phase, timestep):
        shape_new = [int(timestep) + 1, self.nranks]

        if phase not in self.all_rank_msgcnt:
            self.all_rank_msgcnt[phase] = np.zeros(shape_new)
        elif self.all_rank_msgcnt[phase].shape[0] <= timestep:
            self.all_rank_msgcnt[phase].resize(shape_new)
            #  self.all_rank_msgcnt[phase] = np.resize(self.all_rank_msgcnt[phase], shape_new)

        if phase not in self.all_rank_nbrcnt:
            self.all_rank_nbrcnt[phase] = np.zeros(shape_new)
        elif self.all_rank_nbrcnt[phase].shape[0] <= timestep:
            self.all_rank_nbrcnt[phase].resize(shape_new)
            #  self.all_rank_nbrcnt[phase] = np.resize(self.all_rank_nbrcnt[phase], shape_new)

    def Add(self, phase, timestep, src, dest, msgsz):
        self.EnsureFit(phase, timestep)
        try:
            self.all_matrices[phase][timestep][src][dest] += msgsz
        except IndexError as e:
            print('error: ', phase, timestep, src, dest, msgsz)

    def AddMatrix(self, other):
        assert (self.nranks == other.nranks)
        nranks = self.nranks

        def Add2D(a, b):
            a_sh = a.shape
            b_sh = b.shape
            sum_sh = [max(a_sh[0], b_sh[0]), a_sh[1]]
            assert (a_sh[1] == b_sh[1])
            return np.resize(a, sum_sh) + np.resize(b, sum_sh)

        def Add3D(a, b):
            a_sh = a.shape
            b_sh = b.shape
            sum_sh = [max(a_sh[0], b_sh[0]), a_sh[1], a_sh[2]]
            assert (a_sh[1] == b_sh[1])
            assert (a_sh[2] == b_sh[2])
            return np.resize(a, sum_sh) + np.resize(b, sum_sh)

        def GetDefault2D(d, key):
            if key in d:
                return d[key]
            else:
                dflt_2d = np.zeros([1, nranks], dtype=np.int64)
                return dflt_2d

        def GetDefault3D(d, key):
            if key in d:
                return d[key]
            else:
                dflt_3d = np.zeros([1, nranks, nranks], dtype=np.int64)
                return dflt_3d

        all_phases = list(set(list(self.all_matrices.keys()) + list(other.all_matrices.keys())))
        for phase in all_phases:
            # copy all_matrices[phase][timestep][nranks][nranks]
            mat_a = GetDefault3D(self.all_matrices, phase)
            mat_b = GetDefault3D(other.all_matrices, phase)
            self.all_matrices[phase] = Add3D(mat_a, mat_b)

            # copy all_rank_msgcnt[phase][timestep][nranks]
            mat_a = GetDefault2D(self.all_rank_msgcnt, phase)
            mat_b = GetDefault2D(other.all_rank_msgcnt, phase)
            self.all_rank_msgcnt[phase] = Add2D(mat_a, mat_b)

            # copy all_rank_nbrcnt[phase][timestep][nranks]
            mat_a = GetDefault2D(self.all_rank_nbrcnt, phase)
            mat_b = GetDefault2D(other.all_rank_nbrcnt, phase)
            self.all_rank_nbrcnt[phase] = Add2D(mat_a, mat_b)


    def SetMsgCount(self, phase, timestep, src, msgcnt):
        self.EnsureFit2D(phase, timestep)
        self.all_rank_msgcnt[phase][timestep][src] = msgcnt

    def SetNbrCount(self, phase, timestep, src, nbrcnt):
        self.EnsureFit2D(phase, timestep)
        self.all_rank_nbrcnt[phase][timestep][src] = nbrcnt

    def Print(self):
        print(self.all_matrices)

    def Persist(self):
        persisted_state = [self.nranks, self.all_matrices, self.all_rank_msgcnt,
                           self.all_rank_nbrcnt]
        with open('.matrix', 'wb+') as f:
            f.write(pickle.dumps(persisted_state))

    def LoadPersisted(self):
        with open('.matrix', 'rb') as f:
            data_bytes = f.read()
            persisted_state = pickle.loads(data_bytes)
            self.nranks = persisted_state[0]
            self.all_matrices = persisted_state[1]
            self.all_rank_msgcnt = persisted_state[2]
            self.all_rank_nbrcnt = persisted_state[3]

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
        #  all_sums = self.GetPhaseSums()
        #  for phase in all_sums:
        #  print(phase, all_sums[phase])

        print(self.all_rank_msgcnt['LoadBalancing'][13001])

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

    def GetNbrCountStats(self, phase):
        nbrvec = self.all_rank_nbrcnt[phase]
        nbrmean = np.mean(nbrvec, axis=1)
        nbrstd = np.std(nbrvec, axis=1)
        print(nbrmean)
        return nbrmean, nbrstd

    def GetNeighborCountStats(self):
        nbrmean = {}
        nbrstd = {}

        for phase in self.all_rank_nbrcnt:
            phasemean, phasestd = self.GetNbrCountStats(phase)
            phasemean.resize(315)
            phasestd.resize(315)
            nbrmean[phase] = phasemean
            nbrstd[phase] = phasestd

        return nbrmean, nbrstd


def generate_matrix_rank(matrix, dpath, rank):
    rank_csv = "{}/log.{}.csv".format(dpath, rank)
    df = pd.read_csv(rank_csv, on_bad_lines='skip')
    df = df[df['send_or_recv'] == 0]
    df = df[['rank', 'peer', 'timestep', 'phase', 'msg_sz']]

    df_rank_peer = df.groupby(['rank', 'peer', 'timestep', 'phase'],
                              as_index=False).agg(
        ['sum']).reset_index()
    df_rank_peer.columns = df_rank_peer.columns.to_flat_index().str.join('_')
    df_rank_peer.columns = df_rank_peer.columns.str.strip('_')

    df_rank_peer = df_rank_peer.astype(
        {'rank': int, 'peer': int, 'timestep': int, 'phase': str,
         'msg_sz_sum': np.int64})

    for index, row in df_rank_peer.iterrows():
        rank = row['rank']
        peer = row['peer']
        timestep = row['timestep']
        phase = row['phase']
        msgsz = row['msg_sz_sum']

        matrix.Add(phase, timestep, rank, peer, msgsz)

    df_rank = df.groupby(['rank', 'timestep', 'phase'], as_index=False).agg({
        'peer': 'nunique',
        'msg_sz': 'count'
    })

    df_rank = df_rank.astype({'rank': int, 'timestep': int, 'phase': str,
                              'peer': int,
                              'msg_sz': int})

    for index, row in df_rank.iterrows():
        rank = row['rank']
        timestep = row['timestep']
        phase = row['phase']
        msgcnt = row['msg_sz']
        nbrcnt = row['peer']

        #  print(phase, timestep, msgcnt)
        matrix.SetMsgCount(phase, timestep, rank, msgcnt)
        matrix.SetNbrCount(phase, timestep, rank, nbrcnt)


def generate_matrix_rank_min(matrix, dpath, rank):
    rank_csv = "{}.min/log.{}.csv".format(dpath, rank)
    df = pd.read_csv(rank_csv, on_bad_lines='skip')

    df = df.astype({'rank': int, 'peer': int, 'timestep': int, 'phase': str,
                    'msg_sz_count': int})

    #  for index, row in df.iterrows():
        #  rank = row['rank']
        #  peer = row['peer']
        #  timestep = row['timestep']
        #  phase = row['phase']
        #  msgsz = np.int64(row['msg_sz_count'] * row['msg_sz_mean'])

        #  matrix.Add(phase, timestep, rank, peer, msgsz)
    print(df)
    return

    df_rank = df.groupby(['rank', 'timestep', 'phase'], as_index=False).agg({
        'peer': 'nunique',
        'msg_sz_count': 'sum'
    })

    df_rank = df_rank.astype({'rank': int, 'timestep': int, 'phase': str,
                              'peer': int,
                              'msg_sz_count': int})

    for index, row in df_rank.iterrows():
        rank = row['rank']
        timestep = row['timestep']
        phase = row['phase']
        msgcnt = row['msg_sz_count']
        nbrcnt = row['peer']

        #  print(phase, timestep, msgcnt)
        matrix.SetMsgCount(phase, timestep, rank, msgcnt)
        matrix.SetNbrCount(phase, timestep, rank, nbrcnt)


def generate_matrix(dpath):
    nranks = 512

    matrix = CommMatrix(nranks)

    #  for rank in range(nranks):
    for rank in range(4):
        print('Reading Rank {}'.format(rank))
        generate_matrix_rank_min(matrix, dpath, rank)

    #  matrix.Persist()
    #  matrix.LoadPersisted()
    matrix.PrintSummary()


def generate_matrix_parallel(dpath):
    nranks = 512

    matrix = CommMatrix(nranks)

    #  for rank in range(nranks):
    for rank in range(4):
        print('Reading Rank {}'.format(rank))
        matrix_tmp = CommMatrix(nranks)
        generate_matrix_rank_min(matrix_tmp, dpath, rank)
        matrix.AddMatrix(matrix_tmp)

    #  matrix.Persist()
    #  matrix.LoadPersisted()
    matrix.PrintSummary()


def plot_matrix_phasewise_sums(all_sums, fig_dir) -> None:
    fig, ax = plt.subplots(1, 1)

    for phase in all_sums:
        data_y = all_sums[phase]
        data_x = np.arange(len(data_y))
        ax.plot(data_x, data_y, label=phase)

    basefontsz = 18

    ax.legend(prop={'size': basefontsz - 4})
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f} KB'.format(x / 1e6))
    ax.set_xlabel('Timestep', fontsize=basefontsz)
    ax.set_ylabel('Total Communication Per Timestep', fontsize=basefontsz)
    ax.set_title('Phasewise Communication Breakdown\n(512-Rank AMR Run)', fontsize=basefontsz)

    ax.tick_params(axis='x', labelsize=basefontsz - 2)
    ax.tick_params(axis='y', labelsize=basefontsz - 2)

    fig.tight_layout()
    # fig.show()
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
    ax.set_title(
        'Similarity Scores Between Adjacent Comm-Matrices (512-Rank AMR)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('L1 Norm of Comm-Matrix')
    # fig.show()
    fig.tight_layout()
    fig.savefig(fig_dir + '/comm_matrix_sim.pdf', dpi=600)


def plot_matrix_neighbors(nbrmean, nbrstd, fig_path):
    fig, ax = plt.subplots(1, 1)

    phases = list(nbrmean.keys())
    data_x = np.arange(len(nbrmean[phases[0]]))

    for phase in phases:
        ax.plot(data_x, nbrmean[phase], label='Mean {}'.format(phase))

    phase = 'BoundaryComm'
    ax.fill_between(data_x, nbrmean[phase] - nbrstd[phase],
                    nbrmean[phase] + nbrstd[phase],
                    facecolor='green', alpha=0.2,
                    label='+/- 1STD {}'.format(phase))

    phase = 'LoadBalancing'
    ax.fill_between(data_x, nbrmean[phase] - nbrstd[phase],
                    nbrmean[phase] + nbrstd[phase],
                    facecolor='yellow', alpha=0.6,
                    label='+/- 1STD {}'.format(phase))

    phase = 'FluxExchange'
    ax.fill_between(data_x, nbrmean[phase] - nbrstd[phase],
                    nbrmean[phase] + nbrstd[phase],
                    facecolor='purple', alpha=0.2,
                    label='+/- 1STD {}'.format(phase))

    ax.set_title('Mean Neighbor Count For A 512-Rank AMR')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Count')
    ax.legend()

    # fig.show()
    fig.savefig(fig_path + '/neighbor_count.pdf', dpi=600)
    pass


def analyze_matrix():
    nranks = 512
    fig_path = 'figures'

    matrix = CommMatrix(0)
    matrix.LoadPersisted()
    # matrix.PrintSummary()
    matrix_sums = matrix.GetPhaseSums()
    plot_matrix_phasewise_sums(matrix_sums, 'figures')
    # scores = matrix.GetCommMatrixSimilarityScores('BoundaryComm')
    # plot_matrix_sim_scores(scores, fig_path)
    # nbrmean, nbrstd = matrix.GetNeighborCountStats()
    # plot_matrix_neighbors(nbrmean, nbrstd, fig_path)


def run():
    dpath = "/mnt/lt20ad2/parthenon-topo/profile"
    #  generate_matrix(dpath)
    #  generate_matrix_parallel(dpath)
    #  analyze_matrix()


if __name__ == '__main__':
    run()
