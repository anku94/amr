import multiprocessing
import os
import pandas as pd

class TraceMinifier:
    def __init__(self, trace_dir):
        self.trace_dir_ = trace_dir
        self.min_dir_ = trace_dir + '.min'
        try:
            os.mkdir(self.min_dir_)
        except FileExistsError as e:
            pass

    def rank_csv(self, rank):
        csv_path = '{}/log.{}.csv'.format(self.trace_dir_, rank)
        return csv_path

    def rank_min_csv(self, rank):
        csv_path = '{}/log.{}.csv'.format(self.min_dir_, rank)
        return csv_path

    @staticmethod
    def minify_worker(args):
        csv_in = args['csv_in']
        csv_out = args['csv_out']

        print('Processing {} to {}'.format(csv_in, csv_out))

        df_in = pd.read_csv(csv_in)
        df_out = df_in[df_in.send_or_recv == 0].groupby(['rank', 'peer', 'timestep', 'phase'], as_index=False).agg({
            'msg_sz': ['count', 'mean']
        })

        df_out.columns = ['_'.join(col).strip('_') for col in df_out.columns.values]
        df_out.to_csv(csv_out, index=None)

    def minify_args(self, rank):
        csv_in = self.rank_csv(rank)
        csv_out = self.rank_min_csv(rank)
        args = {
            'csv_in': csv_in,
            'csv_out': csv_out
        }
        return args

    def minify(self, rank):
        self.minify_worker(self.minify_args(rank))

    def minify_all(self, num_ranks, num_threads):
        rank_list = list(range(num_ranks))
        minify_args = list(map(lambda x: self.minify_args(x), rank_list))

        with multiprocessing.Pool(processes=num_threads) as p:
            p.map(self.minify_worker, minify_args)


def run():
    trace_dir = '/mnt/lt20ad2/parthenon-topo/profile2'
    tmin = TraceMinifier(trace_dir)
    tmin.minify_all(512, 16)

if __name__ == '__main__':
    run()
