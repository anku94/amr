import multiprocessing
import subprocess
import time
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

    def gen_worker_args(self, rank):
        args = {"rank": rank}

        return args

    @staticmethod
    def worker(args):
        rank = args["rank"]
        print("Worker {}: {}".format(rank, args))
        time.sleep(1)

    def run_worker(self, args):
        rbeg = self.nperrank * self.hidx
        rend = rbeg + self.nperrank

        rend = min(rend, self.nranks)
        self.log("Running workers from {} to {}".format(rbeg, rend))

        all_args = [self.gen_worker_args(rank) for rank in range(rbeg, rend)]

        with multiprocessing.Pool(self.nworkers) as p:
            p.map(self.worker, all_args)

    def log(self, logstr):
        print("h{}: {}".format(self.hidx, logstr))


if __name__ == "__main__":
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    t = Task(trace_dir)
    t.run_worker(None)
