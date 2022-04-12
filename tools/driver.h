//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include <mpi.h>

#include "block.h"
#include "common.h"
#include "topology.h"

class Driver {
 public:
  Driver(const DriverOpts &opts) : opts_(opts) {}

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

  void Run(int argc, char *argv[]) {
    Setup(argc, argv);
    Topology::GenerateMesh(opts_, mesh_);
    mesh_.AllocateBoundaryVariables(opts_.size_per_msg);
    mesh_.Print();
    Destroy();
  }

 private:
  Mesh mesh_;
  const DriverOpts opts_;
};