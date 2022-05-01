import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

def generate_rank(dpath, bins, rank):
    rank_csv = "{}/log.{}.csv".format(dpath, rank)
    df = pd.read_csv(rank_csv, on_bad_lines='skip')
    df = df[df['send_or_recv'] == 0]

    msgsz = df['msg_sz']
    hist, bin_edges = np.histogram(msgsz, bins=bins)
    return hist

def generate(dpath):
    bins = np.arange(0, 10000, 10)
    hist = np.zeros(len(bins) - 1, dtype=int)

    num_ranks = 6
    for rank in range(num_ranks):
        hist += generate_rank(dpath, bins, rank)

    hist_dict = {
        'bins': bins,
        'hist': hist,
    }

    with open('.msgszhist', 'wb+') as f:
        f.write(pickle.dumps(hist_dict))

def read():
    hist_dict = None
    with open('.msgszhist', 'rb') as f:
        hist_dict = pickle.loads(f.read())

    print(hist_dict)

def run():
    dpath = "/mnt/lt20ad2/parthenon-topo/profile"
    generate(dpath)
    #  read()

if __name__ == '__main__':
    run()
