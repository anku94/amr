from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
import ray
import subprocess
import sys
import time
import traceback
from typing import Tuple


def get_node_idx() -> Tuple[int, int]:
    result = subprocess.run(
        ["/share/testbed/bin/emulab-listall"], stdout=subprocess.PIPE
    )
    hoststr = str(result.stdout.decode("ascii"))
    hoststr = hoststr.strip().split(",")
    num_hosts = len(hoststr)
    our_id = open("/var/emulab/boot/nickname", "r").read().split(".")[0][1:]
    our_id = int(our_id)

    return (our_id, num_hosts)


"""
Task module exposes two levels of parallelism:
1. Intra-node: One task is called per-node. It will spawn nworkers threads
   and run worker in each of them.
2. Distributed: One task is called per-cluster. It will spawn a Ray job and
   run the worker in each Ray worker.
"""


class Task:
    nranks = 512
    npernode = 16
    nworkers = 16

    def __init__(self, trace_dir):
        self._trace_dir = trace_dir
        self._bad_nodes = set()
        self._halfwork_nodes = set()

        hidx, nhosts = get_node_idx()
        self.hidx = hidx
        self.nhosts = nhosts

    def gen_worker_fn_args(self):
        return {"trace_dir": self._trace_dir}

    def gen_worker_fn_args_rank(self, rank):
        args = self.gen_worker_fn_args()
        args["rank"] = rank

        return args

    def _get_all_args(self, rbeg=-1, rend=-1):
        if rbeg < 0:
            rbeg = self.npernode * self.hidx

        if rend < 0:
            rend = rbeg + self.npernode

        rend = min(rend, self.nranks)

        self.log("Generating args from {} to {}".format(rbeg, rend))

        all_fn_args = []
        for rank in range(rbeg, rend):
            fn_args = self.gen_worker_fn_args_rank(rank)
            all_fn_args.append(fn_args)

        return all_fn_args

    """ Return all args; for a ray run """

    def run_func_with_ray(self, f):
        all_args = self._get_all_args(0, self.nranks)
        all_futures = [f.remote(args) for args in all_args]
        return ray.get(all_futures)

    def _gen_work_map(self):
        nidx = 0
        rbeg = 0
        rend = 0

        alloc_map = {}

        while rbeg < self.nranks:
            if nidx in self._bad_nodes:
                nidx += 1
                continue

            work_units = self.npernode
            if nidx in self._halfwork_nodes:
                work_units = int(work_units / 2)

            rend = rbeg + work_units
            cur_alloc = rbeg, rend

            alloc_map[nidx] = cur_alloc

            rbeg = rend
            nidx += 1

        return alloc_map

    def skip_node(self, nidx):
        self._bad_nodes.add(nidx)
        pass

    def half_node_alloc(self, nidx):
        self._halfwork_nodes.add(nidx)

    @staticmethod
    def worker(fn_args):
        rank = fn_args["rank"]
        print("Worker {}: {}".format(rank, fn_args))
        time.sleep(1)

    """ Run any single rank, serially, for testing/debugging """

    def run_rank(self, rank):
        rank_args = self.gen_worker_fn_args_rank(rank)
        return self.worker(rank_args)

    """ Run all ranks allocated to node nidx """

    def run_node(self, nidx=-1):
        if nidx == -1:
            nidx, _ = get_node_idx()

        node_work_map = self._gen_work_map()

        if nidx not in node_work_map:
            print("Node {} not allocated any work!".format(nidx))
            return

        rbeg, rend = node_work_map[nidx]
        return self.run_rankwise(rbeg, rend)

    """ Run ranks specified as parameters """

    def run_rankwise(self, rbeg=-1, rend=-1):
        self.log("Running workers from {} to {}".format(rbeg, rend))
        all_fn_args = self._get_all_args(rbeg, rend)
        return self.run_pool(all_fn_args)

    """ Run a set of jobs using Task's thread pool infra """

    def run_pool(self, all_fn_args):
        self.log("Starting pool with {} workers".format(self.nworkers))

        all_data = []

        with ProcessPoolExecutor(max_workers=self.nworkers) as e:
            futures = {e.submit(self.worker, fn_arg): fn_arg for fn_arg in all_fn_args}
            for future in as_completed(futures):
                try:
                    data = future.result()
                    all_data.append(data)
                except Exception as e:
                    print(futures[future])
                    print(e)
                    traceback.print_exc()

            return all_data

    def log(self, logstr):
        print("h{}: {}".format(self.hidx, logstr))


if __name__ == "__main__":
    trace_dir = "/mnt/ltio/parthenon-topo/profile8"
    t = Task(trace_dir)
    t.skip_node(9)
    t.half_node_alloc(10)
    t.half_node_alloc(12)
    m = t._gen_work_map()
    print(m)
    t.run_node(9)
    #  t.run_node(10)
    #  t.run_node(11)
    #  t.run_node(12)
    # t.run_rankwise()
