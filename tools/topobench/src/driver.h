//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include "common.h"
#include "mesh.h"
#include "mesh_gen.h"

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
    if (opts_.topology != NeighborTopology::FromSingleTSTrace and
        opts_.topology != NeighborTopology::FromMultiTSTrace) {
      logvat0(__LOG_ARGS__, LOG_INFO, "[Blocks Per Rank] %zu\n",
              opts_.blocks_per_rank);
      logvat0(__LOG_ARGS__, LOG_INFO, "[Size Per Msg] %zu\n",
              opts_.size_per_msg);
    }

    logvat0(__LOG_ARGS__, LOG_INFO, "[Comm Rounds] %d\n", opts_.comm_rounds);
    logvat0(__LOG_ARGS__, LOG_INFO, "[Topology] %s\n",
            TopologyToStr(opts_.topology).c_str());
    logvat0(__LOG_ARGS__, LOG_INFO, "[Job dir] %s\n", opts_.job_dir);
    logvat0(__LOG_ARGS__, LOG_INFO, "[Log output] %s\n", opts_.bench_log);
  }

  void Run(int argc, char *argv[]) {
    Setup(argc, argv);
    PrintOpts();

    if (opts_.topology == NeighborTopology::FromSingleTSTrace or
        opts_.topology == NeighborTopology::FromMultiTSTrace) {
      RunInternalTrace();
    } else {
      RunInternalNonTrace();
    }

    mesh_.PrintStats();
    Destroy();
  }

  void RunInternalNonTrace() {
    auto mesh_gen = MeshGenerator::Create(opts_);

    int nrounds = mesh_gen->GetNumTimesteps();

    for (int rnum = 0; rnum < nrounds; rnum++) {
      mesh_gen->GenerateMesh(mesh_, rnum);
      mesh_.AllocateBoundaryVariables();
      mesh_.PrintConfig();
      mesh_.DoCommunicationRound();
      mesh_.Reset();
    }
  }

  void RunInternalTrace() {
    auto mesh_gen = MeshGenerator::Create(opts_);

    // int nrounds = topology.GetNumTimesteps();
    const int nts = mesh_gen->GetNumTimesteps();
    const int nts_to_run = std::min(nts, opts_.comm_nts);
    const int nrounds = opts_.comm_rounds;

    logvat0(__LOG_ARGS__, LOG_INFO,
            "num_ts found: %d, num_ts to run: %d, rounds per ts: %d\n"
            "(will skip first ts)",
            nts, nts_to_run, nrounds);

    int ts_beg = 0;
    if (opts_.topology == NeighborTopology::FromMultiTSTrace) {
      // in multi-ts trace, we skip the first timestep
      // as it contains bootstrap communication
      ts_beg = 1;
    }

    for (int ts = ts_beg; ts < nts_to_run; ts++) {
      mesh_gen->GenerateMesh(mesh_, ts);
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
