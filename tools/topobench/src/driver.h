//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include "block.h"
#include "common.h"
#include "topology.h"

#include <mpi.h>

class Driver {
 public:
  Driver(const DriverOpts& opts) : opts_(opts) { Globals::driver_opts = opts; }

  Status Setup(int argc, char* argv[]) {
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
    if (Globals::my_rank == 0) {
      if (opts_.topology != NeighborTopology::FromTrace) {
        printf("[Blocks Per Rank] %zu\n", opts_.blocks_per_rank);
        printf("[Size Per Msg] %zu\n", opts_.size_per_msg);
      }
      printf("[Comm Rounds] %d\n", opts_.comm_rounds);
      printf("[Topology] %s\n", TopologyToStr(opts_.topology).c_str());
    }
  }

  void Run(int argc, char* argv[]) {
    Setup(argc, argv);
    PrintOpts();
    Topology topology(opts_);

    int nrounds = topology.GetNumTimesteps();

    for (int rnum = 0; rnum < nrounds; rnum++) {
      topology.GenerateMesh(opts_, mesh_, rnum);
      mesh_.AllocateBoundaryVariables();
      mesh_.PrintConfig();
      mesh_.DoCommunicationRound();
      mesh_.Reset();
    }

    mesh_.PrintStats();
    Destroy();
  }

 private:
  Mesh mesh_;
  const DriverOpts opts_;
};
