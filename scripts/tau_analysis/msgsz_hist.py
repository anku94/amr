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

    bins = hist_dict['bins']
    bins = np.arange(0, 10000, 1000)
    hist = hist_dict['hist']
    hist_new = []
    for idx, x in enumerate(hist[::100]):
        chunk_sum = sum(hist[idx*100:idx*100 + 100])
        hist_new.append(chunk_sum)

    hist_new = np.array(hist_new, dtype=np.float64)
    hist_new /= sum(hist_new)

    print(bins)
    print(hist_new)

    basefontsz = 18

    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(10) + 0.4, hist_new, width=0.9)
    ax.set_title('Distribution of AMR Message Sizes', fontsize=basefontsz + 2)
    ax.set_xlabel('Message Size', fontsize=basefontsz)
    ax.set_ylabel('Percent Of Messages', fontsize=basefontsz)
    ax.set_xticks(range(10))
    ax.xaxis.set_major_formatter(lambda x, pos: '{:.0f}K'.format(x))
    ax.yaxis.set_major_formatter(lambda x, pos: '{:.0f}%'.format(x*100))
    ax.tick_params(axis='x', labelsize=basefontsz - 2)
    ax.tick_params(axis='y', labelsize=basefontsz - 2)

    fig.tight_layout()
    # fig.show()
    fig.savefig('figures/msgsz_hist.pdf', dpi=600)


def run():
    # dpath = "/mnt/lt20ad2/parthenon-topo/profile"
    # generate(dpath)
    read()


if __name__ == '__main__':
    run()
