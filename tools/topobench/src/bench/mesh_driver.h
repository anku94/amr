#pragma once

#include <glog/logging.h>
#include <mpi.h>

#include <vector>

#include "amr/mesh_utils.h"
#include "bench/comm_mesh.h"
#include "logging.h"

namespace topo {
struct MeshDriverOpts {
  topo::Vec3i mesh_dims;
  int max_reflvl;
  int my_rank = 0;  // local MPI rank
  int nranks = 1;   // total MPI ranks
  std::string policy = "baseline";
  std::string jobdir = "/tmp";
};

class MeshDriver {
 public:
  MeshDriver(const MeshDriverOpts& opts);

  void PrintOpts() {
    MLOG(MLOG_INFO, "Mesh dims: %s", opts_.mesh_dims.ToString().c_str());
    MLOG(MLOG_INFO, "Max reflvl: %d", opts_.max_reflvl);
    MLOG(MLOG_INFO, "My rank: %d", opts_.my_rank);
    MLOG(MLOG_INFO, "Total ranks: %d", opts_.nranks);
  }

  void Run();

  void RunWithOmesh(OrderedMesh& omesh);

  // AssignBlocks: populate ranklist using a synthetic costlist
  // generated using distribution, + placement scheme in opts.policy
  int AssignBlocks(std::vector<int>& ranklist, int nblocks, int nranks);

 private:
  const MeshDriverOpts opts_;
  CommMesh comm_mesh_;
};
}  // namespace topo
