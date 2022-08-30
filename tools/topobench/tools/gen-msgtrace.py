import pandas as pd
import random


def gen_empty_df():
    cols = [
        "rank",
        "peer",
        "timestep",
        "phase",
        "msg_id",
        "send_or_recv",
        "msg_sz",
        "timestamp",
    ]
    return pd.DataFrame(columns=cols)


def init_ds(nranks):
    ds = [[] for r in range(nranks)]
    return ds


def add_row_ds(ds, rank, peer, ts, phase, msg_id, sr, msg_sz):
    row = [rank, peer, ts, phase, msg_id, sr, msg_sz, 147001]
    ds[rank].append(row)


def add_send_ds(ds, rank, peer, ts, phase, msg_id, msg_sz):
    add_row_ds(ds, rank, peer, ts, phase, msg_id, 0, msg_sz)
    add_row_ds(ds, peer, rank, ts, phase, msg_id, 1, msg_sz)


def add_bcomm_ds(ds, ts, rank_a, rank_b, msgsz_a, msgsz_b):
    add_send_ds(ds, rank_a, rank_b, ts, "BoundaryComm", rank_a, msgsz_a)
    add_send_ds(ds, rank_b, rank_a, ts, "BoundaryComm", rank_b, msgsz_b)


def write_allds_as_df(ds, dpath):
    cols = [
        "rank",
        "peer",
        "timestep",
        "phase",
        "msg_id",
        "send_or_recv",
        "msg_sz",
        "timestamp",
    ]

    for rank, rank_ds in enumerate(ds):
        rank_df = pd.DataFrame.from_records(rank_ds, columns=cols)
        rank_df_path = "{}/msgs.{}.csv".format(dpath, rank)
        print("Writing {}-shaped table to {}...".format(rank_df.shape, rank_df_path))
        rank_df.to_csv(rank_df_path, sep="|", index=None)


def add_ts_ring(ds, ts):
    nranks = len(ds)
    for rank_a in range(nranks):
        rank_b = (rank_a + ts + 1) % nranks
        msgsz_a = 512 * random.randint(1, 4)
        msgsz_b = 512 * random.randint(1, 4)
        print(
            "TS{}: Adding {}->{} (msgsz: {}/{})".format(
                ts, rank_a, rank_b, msgsz_a, msgsz_b
            )
        )
        add_bcomm_ds(ds, ts, rank_a, rank_b, msgsz_a, msgsz_b)


def gen_msgs_ring(nranks, nts, trace_path):
    ds = init_ds(nranks)

    for cur_ts in range(nts):
        print("---")
        add_ts_ring(ds, cur_ts)

    print("---")
    write_allds_as_df(ds, trace_path)


def run():
    nranks = 4
    nts = 2
    trace_path = "msgs"
    gen_msgs_ring(nranks, nts, trace_path)


if __name__ == "__main__":
    run()
