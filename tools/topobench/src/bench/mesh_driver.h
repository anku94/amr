#pragma once

#include <glog/logging.h>
#include <mpi.h>

#include "amr/mesh.h"
#include "amr/mesh_utils.h"
#include "amr/print_utils.h"
#include "amr/globals.h"
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
    MLOG(MLOG_INFO, "Mesh dims: %s", opts_.mesh_dims.ToString().c_str());
    MLOG(MLOG_INFO, "Max reflvl: %d", opts_.max_reflvl);
  }

  void Run() {
    MLOG(MLOG_INFO, "Running mesh driver");
    auto dims = opts_.mesh_dims;
    auto lvl = opts_.max_reflvl;

    amr::Mesh mesh(dims.x, dims.y, dims.z, lvl);
    amr::PrintUtils::PrintHierarchy(mesh, amr::Mesh::GetRootLoc());

    auto omesh = mesh.GetOrderedMesh();
    amr::PrintUtils::PrintOmesh(omesh);
  }

 private:
  const MeshDriverOpts opts_;
};
}  // namespace topo::bench
