import collections
import numpy as np
import pandas as pd
import re
import unittest

from operator import itemgetter
from plot_msgs import to_dense_2d
from typing import List

"""
Interesting files:
- /aggregate.csv
ts,evtname,evtval
- (renamed) /trace/phases.aggr.by_ts.csv
- /aggr/msg_concat.csv
timestep,phase,send_or_recv,rank,msg_sz_count
- /run/logstats.csv - from analyze_taskflow.py
cycle,time,dt,zone-cycles/wsec_step,wsec_total,wsec_step,zone-cycles/wsec,wsec_AMR
- /run/log.txt.csv - from scripts/phoebus-runs.sh
cycle,time,dt,zc_per_step,wtime_total,wtime_step_other,zc_wamr,wtime_step_amr 


Other files:

- /aggr/msgs.R.csv - aggregate msg stats per rank+ts+phase
- /phases/phases.R.csv -  rank,ts,evtname,evtval
- /traces/funcs.R.csv
- /traces/msgs.R.csv
- /traces/state.R.csv
"""


class TraceReader:
    def __init__(self, dir_path):
        print(f"[TraceReader::Init] {dir_path}")

        self._dir_path = dir_path
        self._df_cache = {}

    def _read_file(self, fpath: str, sep=",", cache=True) -> pd.DataFrame:
        full_path = "{}/{}".format(self._dir_path, fpath)

        if full_path in self._df_cache:
            return self._df_cache[full_path]

        df = pd.read_csv(full_path, sep=sep)

        if cache:
            self._df_cache[full_path] = df

        return df

    def read_aggr_rw(self):
        aggr_path = "trace/phases.aggr.csv"
        return self._read_file(aggr_path)

    def get_aggr_rw_key(self, key: str):
        aggr_df = self.read_aggr_rw()
        row = aggr_df[aggr_df["evtname"] == key]["evtval"].iloc[0]
        row = np.array([int(i) for i in row.split(",")], dtype=np.int64)

        print(f"Reading {key}: {len(row)} items retrieved")

        return row

    def read_aggr_tsw(self):
        aggr_path = "trace/phases.aggr.by_ts.csv"
        return self._read_file(aggr_path)

    def get_aggr_tsw_key(self, key: str):
        aggr_df = self.read_aggr_tsw()
        row = aggr_df[aggr_df["evtname"] == key]["evtval"].iloc[0]
        row = np.array([int(i) for i in row.split(",")], dtype=np.int64)

        print(f"Reading {key}: {len(row)} items retrieved")

        return row

    def read_msg_concat(self):
        msg_concat_path = "aggr/msg_concat.csv"
        df_msg = self._read_file(msg_concat_path)
        df_msg = df_msg.astype(
            {
                "timestep": int,
                "phase": str,
                "send_or_recv": int,
                "rank": str,
                "msg_sz_count": str,
            }
        )

        print(f"Reading msg_concat: {len(row)} items retrieved")

        return df_msg

    def read_rank_trace(self, rank: int) -> pd.DataFrame:
        trace_path = "trace/funcs/funcs.{}.csv".format(rank)
        return self._read_file(trace_path, sep="|")

    """
    All ranks emit identical state values, so any arbitrary rank can be read
    """

    def read_tau_state(self, rank=0):
        state_path = "trace/state/state.{}.csv".format(rank)
        return self._read_file(state_path, sep="|")

    def read_logstats(self):
        logstats_path = "run/log.txt.csv"
        return self._read_file(logstats_path)

    def get_tau_event(self, event: str) -> None:
        df = self.read_aggregate()
        df_rel = df[df["evtname"] == event]
        evt_vals = TraceReader._parse_arrays(df_rel["evtval"])

        print(f"Reading {event}: {len(evt_vals)} items retrieved")

        return evt_vals

    def get_tau_state_(self, state: str) -> pd.DataFrame:
        df = self.read_tau_state()
        df_key = df[df["key"] == state].copy()

        def strtolsi(x):
            ls_str = x.strip(",").split(",")
            return list(map(int, ls_str))

        df_key["val"] = df_key["val"].apply(strtolsi)

        print(f"Reading {state}: {len(df_key)} items retrieved")

        return df_key

    def get_rank_alloc(self) -> None:
        alloc_df = self.get_tau_state_("RL")

        nranks = max(alloc_df["val"].iloc[0]) + 1

        all_allocs = []
        prev_alloc = [0] * nranks
        prev_idx = 0

        for row in alloc_df.itertuples():
            cur_ts = row[2]
            cur_ranks = row[4]
            cur_rcnt_items = collections.Counter(cur_ranks).items()
            cur_rcnt = list(map(itemgetter(1), sorted(cur_rcnt_items)))

            assert len(cur_rcnt) == nranks

            while prev_idx < cur_ts:
                all_allocs.append(prev_alloc)
                prev_idx += 1

            all_allocs.append(cur_rcnt)
            prev_alloc = cur_rcnt

        return all_allocs

    def get_msg_count(self, event: str) -> None:
        df = self.read_msg_concat()
        df = df[df["phase"] == event]
        nsteps = df["timestep"].max()
        msgcnt_key = "msg_sz_count"
        dense_mat = TraceReader._to_dense_2d(df, msgcnt_key, nsteps, nranks=512)
        return dense_mat

    def get_msg_sz(self, event: str) -> None:
        df = self.read_msg_concat()
        df = df[df["phase"] == event]
        nsteps = df["timestep"].max()
        msgsz_key = "msg_sz_mean"
        dense_mat = TraceReader._to_dense_2d(df, msgsz_key, nsteps, nranks=512)
        return dense_mat

    def get_msg_npeers(self, event: str) -> None:
        df = self.read_msg_concat()
        df = df[df["phase"] == event]
        nsteps = df["timestep"].max()
        npeer_key = "peer_nunique"
        dense_mat = TraceReader._to_dense_2d(df, npeer_key, nsteps, nranks=512)
        return dense_mat

    @staticmethod
    def _parse_arrays(df_col: pd.Series) -> None:
        def parse_array(arr):
            arr = arr.strip("[]").split(",")
            arr = [int(i) for i in arr]
            return arr

        parsed_col = df_col.map(parse_array)
        parsed_col_lens = parsed_col.map(len)
        len_counters = collections.Counter(parsed_col_lens)
        len_common = len_counters.most_common(1)[0][0]

        idxes_to_drop = [x[0] for x in enumerate(parsed_col_lens) if x[1] != len_common]

        parsed_col_dropped = [
            x[1] for x in enumerate(parsed_col) if x[0] not in idxes_to_drop
        ]

        print("Common len: ", len_common)
        print("Dropping indices: {}".format(",".join([str(i) for i in idxes_to_drop])))
        print("Len Old: {}, New: {}".format(len(parsed_col), len(parsed_col_dropped)))

        evtvals = np.array(parsed_col_dropped)
        print("Nparray Shape: ", evtvals.shape)

        return evtvals

    @staticmethod
    def _to_dense(idx, cnt, nranks=512):
        dense_arr = []

        csrint = lambda x: [int(i) for i in x.split(",")]
        keys = csrint(idx)
        vals = csrint(cnt)
        sdict = dict(zip(keys, vals))

        for i in range(nranks):
            i_val = 0
            if i in sdict:
                i_val = sdict[i]
            dense_arr.append(i_val)

        return np.array(dense_arr)

    @staticmethod
    def _to_dense_2d(df, key, nts, nranks=512):
        all_rows = {}

        for idx, row in df.iterrows():
            ts = row["timestep"]
            idxes = row["rank"]
            #  counts = row["msg_sz_count"]
            counts = row[key]
            row_dense = TraceReader._to_dense(idxes, counts, nranks)

            if ts in all_rows:
                all_rows[ts] += row_dense
            else:
                all_rows[ts] = row_dense

        all_rows_dense = []

        for ts in range(nts):
            ts_row = np.zeros(nranks, dtype=int)
            if ts in all_rows:
                ts_row = all_rows[ts]

            all_rows_dense.append(ts_row)

        mat_2d = np.stack(all_rows_dense, axis=0)
        return mat_2d


class TraceOps:
    def __init__(self, dir_path):
        self.trace = TraceReader(dir_path)

    @classmethod
    def split_eqn(cls, eqn):
        symbols = re.split("\+|-", eqn)

        def sign(eqn, symbol):
            idx = eqn.find(symbol)
            symbol = "+"
            if idx > 0:
                symbol = eqn[idx - 1]
            return symbol

        pos = [s for s in symbols if sign(eqn, s) == "+"]
        neg = [s for s in symbols if sign(eqn, s) == "-"]
        return [pos, neg]

    @classmethod
    def _parse_query(cls, query_str) -> List:
        match = re.fullmatch(r"(^.*?):(.*$)", query_str)
        pref, rest = match.group(1), match.group(2)
        if ":" in rest:
            return [pref] + cls._parse_query(rest)
        else:
            return [pref, cls.split_eqn(rest)]

    @classmethod
    def cropsum_2d(cls, mats):
        len_min = min([m.shape[0] for m in mats])
        mats = [m[:len_min] for m in mats]
        #  print([m.shape for m in mats])
        mat_agg = np.sum(mats, axis=0)
        #  print(mat_agg.shape)
        return mat_agg

    @classmethod
    def multimat_labels(cls, labels, f):
        labels_pos = labels[0]
        labels_neg = labels[1]

        mats_pos = [f(l) for l in labels_pos]
        mat_pos_agg = cls.cropsum_2d(mats_pos)

        if len(labels_neg) > 0:
            mats_neg = [f(l) for l in labels_neg]
            mat_neg_agg = cls.cropsum_2d(mats_neg)

            return mat_pos_agg - mat_neg_agg

        return mat_pos_agg

    # XXX: some memory usage issues possible?
    def multimat(self, query_str):
        parsed_query = self._parse_query(query_str)
        qtype = parsed_query[:-1]
        qlabels = parsed_query[-1]

        if qtype[0] == "rcnt":
            return np.array(self.trace.get_rank_alloc())

        f = None

        if qtype[0] == "tau":
            f = self.trace.get_tau_event
        elif qtype[0] == "msgcnt":
            f = self.trace.get_msg_count
        elif qtype[0] == "msgsz":
            f = self.trace.get_msg_sz
        elif qtype[0] == "npeer":
            f = self.trace.get_msg_npeers
        elif qtype[0] == "aggr" and qtype[1] == "rw":
            f = self.trace.get_aggr_rw_key
        elif qtype[0] == "aggr" and qtype[1] == "tsw":
            f = self.trace.get_aggr_tsw_key
        else:
            assert False

        return self.multimat_labels(qlabels, f)


def TraceReaderTest():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    tr = TraceReader(trace_dir)
    #  print(tr.read_logstats())
    #  tr.get_tau_event("AR2")
    tr.get_tau_state("RL")


class TestTraceReader(unittest.TestCase):
    def test_parse_query(self):
        query_str = "tau:AR3"
        parsed_query = TraceOps._parse_query(query_str)
        self.assertEqual(len(parsed_query), 2)
        self.assertEqual(parsed_query[0], "tau")
        self.assertEqual(parsed_query[1][0][0], "AR3")

        query_str = "aggr:rw:AR3-AR3U"
        parsed_query = TraceOps._parse_query(query_str)
        self.assertEqual(len(parsed_query), 3)
        self.assertEqual(parsed_query[0], "aggr")
        self.assertEqual(parsed_query[1], "rw")
        self.assertEqual(parsed_query[2][0][0], "AR3")
        self.assertEqual(parsed_query[2][1][0], "AR3U")

    def test_read_aggr(self):
        trace_path = "/mnt/ltio/parthenon-topo/profile14"
        tr = TraceOps(trace_path)
        data = tr.multimat("aggr:rw:AR3-AR3_UMBT")
        assert len(data) == 512


if __name__ == "__main__":
    unittest.main()
