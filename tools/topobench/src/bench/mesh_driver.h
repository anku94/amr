#pragma once

#include <glog/logging.h>
#include <mpi.h>

#include "amr/mesh_utils.h"
#include "globals.h"
#include "mesh.h"
#include "topo_common.h"

namespace topo::bench {
struct MeshDriverOpts {
  topo::amr::Vec3i mesh_dims;
  int max_reflvl;
};

class MeshDriver {
 public:
  MeshDriver(const MeshDriverOpts &opts) : opts_(opts) { PrintOpts(); }

  void PrintOpts() {
    MLOG(LOG_INFO, "Mesh dims: %s", opts_.mesh_dims.ToString().c_str());
    MLOG(LOG_INFO, "Max reflvl: %d", opts_.max_reflvl);
  }

 private:
  const MeshDriverOpts opts_;
};
}  // namespace topo::bench
