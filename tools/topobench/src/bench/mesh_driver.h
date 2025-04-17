#pragma once

#include <glog/logging.h>
#include <mpi.h>

#include <vector>

#include "amr/mesh_utils.h"
#include "logging.h"

namespace topo::bench {
struct MeshDriverOpts {
  topo::amr::Vec3i mesh_dims;
  int max_reflvl;
  int my_rank = 0;  // local MPI rank
  int nranks = 1;   // total MPI ranks
};

struct BlockPlacement {
  int nblocks;
  int nranks;
  std::vector<double> costlist;
  std::vector<double> ranklist;
};

class MeshDriver {
 public:
  MeshDriver(const MeshDriverOpts &opts) : opts_(opts) { PrintOpts(); }

  void PrintOpts() {
    MLOG(MLOG_INFO, "Mesh dims: %s", opts_.mesh_dims.ToString().c_str());
    MLOG(MLOG_INFO, "Max reflvl: %d", opts_.max_reflvl);
    MLOG(MLOG_INFO, "My rank: %d", opts_.my_rank);
    MLOG(MLOG_INFO, "Total ranks: %d", opts_.nranks);
  }

  void Run();

 private:
  const MeshDriverOpts opts_;
};
}  // namespace topo::bench
