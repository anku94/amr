from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
import subprocess
import sys
import time
import traceback
from typing import Tuple

"""
Task module exposes two levels of parallelism:
1. Intra-node: One task is called per-node. It will spawn nworkers threads
   and run worker in each of them.
2. Distributed: One task is called per-cluster. It will spawn a Ray job and
   run the worker in each Ray worker.
"""


class Task:
    nranks = 512
    nperrank = 16
    nworkers = 16

    def __init__(self, trace_dir):
        self._trace_dir = trace_dir

        hidx, nhosts = self.get_node_idx()
        self.hidx = hidx
        self.nhosts = nhosts

    @classmethod
    def get_node_idx(cls) -> Tuple[int, int]:
        result = subprocess.run(
            ["/share/testbed/bin/emulab-listall"], stdout=subprocess.PIPE
        )
        hoststr = str(result.stdout.decode("ascii"))
        hoststr = hoststr.strip().split(",")
        num_hosts = len(hoststr)
        our_id = open("/var/emulab/boot/nickname", "r").read().split(".")[0][1:]
        our_id = int(our_id)

        return (our_id, num_hosts)

    def gen_worker_fn_args(self, rank):
        fn_args = {"rank": rank}

        return fn_args

    @staticmethod
    def worker(fn_args):
        rank = fn_args["rank"]
        print("Worker {}: {}".format(rank, fn_args))
        time.sleep(1)

    def run_worker(self):
        rbeg = self.nperrank * self.hidx
        rend = rbeg + self.nperrank

        rend = min(rend, self.nranks)
        self.log("Running workers from {} to {}".format(rbeg, rend))

        base_fn_args = self.gen_worker_fn_args()
        all_fn_args = []
        for rank in range(rbeg, rend):
            fn_args = copy.deepcopy(base_fn_args)
            fn_args["rank"] = rank
            all_fn_args.append(fn_args)

        self.log("Starting pool with {} workers".format(self.nworkers))

        #  with multiprocessing.Pool(processes=self.nworkers) as p:
        #  p.map(self.worker, all_fn_args)
        with ProcessPoolExecutor(max_workers=self.nworkers) as e:
            #  ret = e.map(self.worker, all_fn_args)
            #  print(ret)
            futures = {e.submit(self.worker, fn_arg): fn_arg for fn_arg in all_fn_args}
            for future in as_completed(futures):
                try:
                    data = future.result()
                    print(data)
                except Exception as e:
                    print(futures[future])
                    print(e)
                    traceback.print_exc()

    def log(self, logstr):
        print("h{}: {}".format(self.hidx, logstr))


if __name__ == "__main__":
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    t = Task(trace_dir)
    t.run_worker()
