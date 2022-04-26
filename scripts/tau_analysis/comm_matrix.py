import numpy as np
import pandas as pd
import pickle

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



def generate_matrix_rank(matrix, dpath, rank):
    rank_csv = "{}/log.{}.csv".format(dpath, rank)
    df = pd.read_csv(rank_csv, on_bad_lines='skip')
    df = df[df['send_or_recv'] == 0]
    df = df[['rank', 'peer', 'timestep', 'phase', 'msg_sz']]
    df = df.groupby(['rank', 'peer', 'timestep', 'phase'], as_index=False).agg(['sum']).reset_index()
    df.columns = df.columns.to_flat_index().str.join('_')
    df.columns = df.columns.str.strip('_')

    df = df.astype({'rank': int, 'peer': int, 'timestep': int, 'phase': str, 'msg_sz_sum': np.int64})

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


def analyze_matrix():
    nranks = 512

    matrix = CommMatrix(0)
    matrix.LoadPersisted()
    matrix.PrintSummary()

def run():
    dpath = "/mnt/lt20ad2/parthenon-topo/profile"
    #  generate_matrix(dpath)
    analyze_matrix()

if __name__ == '__main__':
    run()
