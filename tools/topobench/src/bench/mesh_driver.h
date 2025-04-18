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
  int my_rank;         // local MPI rank
  int nranks;          // total MPI ranks
  std::string policy;  // placement policy
  std::string jobdir;  // job directory
  std::string log_fname;  // log file name
  int num_ts;          // num timesteps
  int num_rounds;      // num rounds/timestep
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

  // Run: Run for num_ts timesteps
  void Run();

  // RunWithOmesh: Run with an ordered mesh for nrounds
  // (with a single mesh)
  void RunWithOmesh(OrderedMesh& omesh, std::vector<int>& ranklist);

  // AssignBlocks: populate ranklist using a synthetic costlist
  // generated using distribution, + placement scheme in opts.policy
  int AssignBlocks(std::vector<int>& ranklist, int nblocks, int nranks);

 private:
  const MeshDriverOpts opts_;
  CommMesh comm_mesh_;

  std::string GetLogPath() const {
    return opts_.jobdir + "/" + opts_.log_fname;
  }
};
}  // namespace topo
