import collections
import numpy as np
import pandas as pd

"""
Interesting files:
- /aggregate.csv
ts,evtname,evtval
- /aggr/msg_concat.csv
timestep,phase,send_or_recv,rank,msg_sz_count
- /run/logstats.csv
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

    def read_aggregate(self):
        aggr_path = "aggregate.csv"
        return self._read_file(aggr_path)

    def read_msg_concat(self):
        msg_concat_path = "aggr/msg_concat.csv"
        return self._read_file(msg_concat_path)

    def read_logstats(self):
        logstats_path = "run/log.txt.csv"
        return self._read_file(logstats_path)

    def get_tau_event(self, event: str) -> None:
        df = self.read_aggregate()
        df_rel = df[df["evtname"] == event]
        evt_vals = TraceReader._parse_arrays(df_rel["evtval"])
        return evt_vals

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
        print('Nparray Shape: ', evtvals.shape)

        return parsed_col_dropped


def TraceReaderTest():
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    tr = TraceReader(trace_dir)
    #  print(tr.read_logstats())
    tr.get_tau_event("AR2")


if __name__ == "__main__":
    TraceReaderTest()
