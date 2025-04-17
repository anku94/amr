#include "mesh_driver.h"

#include "amr/block.h"
#include "amr/mesh.h"
#include "amr/mesh_utils.h"
#include "amr/print_utils.h"
#include "amr_lb.h"
#include "bench/comm_mesh.h"
#include "bench/placement_utils.h"
#include "topo_common.h"

using PlacementArgs = ::amr::lb::PlacementArgs;
using LoadBalance = ::amr::lb::LoadBalance;
using MeshBlockRef = std::shared_ptr<topo::amr::MeshBlock>;

namespace topo::bench {
void MeshDriver::Run() {
  MLOG(MLOG_INFO, "Running mesh driver");
  auto dims = opts_.mesh_dims;
  auto lvl = opts_.max_reflvl;

  // Create base mesh and print it
  amr::Mesh mesh(dims.x, dims.y, dims.z, lvl);
  amr::PrintUtils::PrintHierarchy(mesh, amr::Mesh::GetRootLoc());

  // Generate ordered mesh and print it
  auto omesh = mesh.GetOrderedMesh();
  amr::PrintUtils::PrintOmesh(omesh);

  // Compute load-balanced placement
  auto nblocks = omesh.nblocks;
  std::vector<double> costlist(nblocks, 1.0);
  std::vector<int> ranklist(nblocks, 0);
  int nranks = opts_.nranks;

  amr::Globals::my_rank = opts_.my_rank;
  amr::Globals::nranks = opts_.nranks;

  PlacementArgs lb_args{"baseline", costlist, ranklist, nranks};
  int rv = LoadBalance::AssignBlocks(lb_args);
  ABORTIF(!rv, "Placement assignment failed!");
  MLOG(MLOG_INFO, "Placement assignment complete.");

//   CommMesh comm_mesh;
//   rv = PlacementUtils::SetupCommMesh(comm_mesh, omesh, lb_args.ranklist);
//   MLOGIF(!rv, MLOG_ERRO, "SetupCommMesh failed!");

//   // Allocate boundary variables for communication
//   auto s = comm_mesh.AllocateBvars();
//   MLOGIF(s != Status::OK, MLOG_ERRO, "Failed to allocate boundary variables!");

//   MLOG(MLOG_INFO, "Allocated boundary variables");

//   // Run one communication round
//   s = comm_mesh.DoCommunicationRound();
//   MLOGIF(s != Status::OK, MLOG_ERRO, "Communication round failed!");
//   MLOG(MLOG_INFO, "Communication round complete.");

//   // Print stats and cleanup
//   comm_mesh.PrintStats();
//   comm_mesh.ResetBvarsAndBlocks();
}
}  // namespace topo::bench
