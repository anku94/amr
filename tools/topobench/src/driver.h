//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include "block.h"
#include "common.h"
#include "mesh.h"
#include "topology.h"

#include <mpi.h>

class Driver {
public:
  Driver(const DriverOpts &opts) : opts_(opts) { Globals::driver_opts = opts; }

  Status Setup(int argc, char *argv[]) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
      return Status::MPIError;
    }

    if (MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank)) != MPI_SUCCESS) {
      return Status::MPIError;
    }

    if (MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks) != MPI_SUCCESS) {
      return Status::MPIError;
    }

    return Status::OK;
  }

  Status Destroy() {
    MPI_Finalize();
    return Status::OK;
  }

  void PrintOpts() {
    if (opts_.topology != NeighborTopology::FromTrace) {
      logvat0(__LOG_ARGS__, LOG_INFO, "[Blocks Per Rank] %zu\n",
              opts_.blocks_per_rank);
      logvat0(__LOG_ARGS__, LOG_INFO, "[Size Per Msg] %zu\n",
              opts_.size_per_msg);
    }

    logvat0(__LOG_ARGS__, LOG_INFO, "[Comm Rounds] %d\n", opts_.comm_rounds);
    logvat0(__LOG_ARGS__, LOG_INFO, "[Topology] %s\n",
            TopologyToStr(opts_.topology).c_str());
    logvat0(__LOG_ARGS__, LOG_INFO, "[Log output] %s\n", opts_.bench_log);
  }

  void Run(int argc, char *argv[]) {
    Setup(argc, argv);
    PrintOpts();

    if (opts_.topology == NeighborTopology::FromTrace) {
      RunInternalTrace();
    } else {
      RunInternalNonTrace();
    }

    mesh_.PrintStats();
    Destroy();
  }

  void RunInternalNonTrace() {
    Topology topology(opts_);

    int nrounds = topology.GetNumTimesteps();

    for (int rnum = 0; rnum < nrounds; rnum++) {
      topology.GenerateMesh(opts_, mesh_, rnum);
      mesh_.AllocateBoundaryVariables();
      mesh_.PrintConfig();
      mesh_.DoCommunicationRound();
      mesh_.Reset();
    }
  }

  void RunInternalTrace() {
    Topology topology(opts_);

    // int nrounds = topology.GetNumTimesteps();
    const int nts = topology.GetNumTimesteps();
    const int nts_to_run = std::min(nts, opts_.comm_nts);
    const int nrounds = opts_.comm_rounds;

    logvat0(__LOG_ARGS__, LOG_INFO,
            "num_ts found: %d, num_ts to run: %d, rounds per ts: %d\n"
            "(will skip first ts)",
            nts, nts_to_run, nrounds);

    // first ts does init comm that is not reflective of other rounds
    // we start from ts=1
    for (int ts = 1; ts < nts_to_run; ts++) {
      topology.GenerateMesh(opts_, mesh_, ts);
      MPI_Barrier(MPI_COMM_WORLD);

      mesh_.AllocateBoundaryVariables();
      mesh_.PrintConfig();

      for (int rnum = 0; rnum < nrounds; rnum++) {
        mesh_.DoCommunicationRound();
        MPI_Barrier(MPI_COMM_WORLD);
      }

      mesh_.Reset();
    }
  }

private:
  Mesh mesh_;
  const DriverOpts opts_;
};
