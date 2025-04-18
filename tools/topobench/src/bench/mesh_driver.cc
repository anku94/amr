#include "mesh_driver.h"

#include "amr/block.h"
#include "amr/mesh.h"
#include "amr/mesh_utils.h"
#include "amr/print_utils.h"
#include "amr_lb.h"
#include "bench/comm_mesh.h"
#include "bench/placement_utils.h"
#include "distributions.h"
#include "logging.h"

using MeshBlockRef = std::shared_ptr<topo::MeshBlock>;
using DistributionUtils = amr::DistributionUtils;

namespace topo {
MeshDriver::MeshDriver(const MeshDriverOpts& opts) : opts_(opts) {
  Globals::my_rank = opts_.my_rank;
  Globals::nranks = opts_.nranks;
  PrintOpts();
}


void MeshDriver::Run() {
  MLOG(MLOG_INFO, "Running mesh driver");
  auto dims = opts_.mesh_dims;
  auto lvl = opts_.max_reflvl;

  // Create base mesh and print it
  Mesh mesh(dims.x, dims.y, dims.z, lvl);
  PrintUtils::PrintHierarchy(mesh, Mesh::GetRootLoc());

  // Generate ordered mesh and print it
  auto omesh = mesh.GetOrderedMesh();
  PrintUtils::PrintOmesh(omesh);

  RunWithOmesh(omesh);
}

void MeshDriver::RunWithOmesh(OrderedMesh& omesh) {
  // Compute load-balanced placement
  int nblocks = omesh.nblocks;
  std::vector<int> ranklist(nblocks, -1);
  int rv = AssignBlocks(ranklist, nblocks, opts_.nranks);
  ABORTIF(rv, "Placement assignment failed!");

  rv = PlacementUtils::SetupCommMesh(comm_mesh_, omesh, ranklist);
  MLOGIF(rv, MLOG_ERRO, "SetupCommMesh failed!");

  // Allocate boundary variables for communication
  auto s = comm_mesh_.AllocateBvars();
  MLOGIF(s != Status::OK, MLOG_ERRO, "Failed to allocate boundary variables!");
  MLOG(MLOG_INFO, "Allocated boundary variables");

  // Run one communication round
  s = comm_mesh_.DoCommunicationRound();
  MLOGIF(s != Status::OK, MLOG_ERRO, "Communication round failed!");
  MLOG(MLOG_INFO, "Communication round complete.");

  //   // Print stats and cleanup
  comm_mesh_.PrintStats();
  comm_mesh_.ResetBvarsAndBlocks();
}

int MeshDriver::AssignBlocks(std::vector<int>& ranklist, int nblocks,
                             int nranks) {
  std::vector<double> costlist(nblocks, -1.0);
  auto dopts = DistributionUtils::GetConfigOpts();
  DistributionUtils::GenDistribution(dopts, costlist, nblocks);

  MLOGIFR0(MLOG_INFO, "AssignBlocks: gen costs for %d blocks using distrib: %s",
       nblocks, DistributionUtils::DistributionOptsToString(dopts).c_str());

  auto policy = opts_.policy.c_str();
  MLOGIFR0(MLOG_INFO, "AssignBlocks: using policy %s", policy);

  auto lb_args = amr::lb::PlacementArgs{policy, costlist, ranklist, nranks};
  int rv = amr::lb::LoadBalance::AssignBlocks(lb_args);

  ABORTIF(rv, "Placement assignment failed!");

  auto coststr = PrintUtils::SerializeVec(costlist);
  MLOGIFR0(MLOG_INFO, "AssignBlocks: costlist %s", coststr.c_str());

  auto rankstr = PrintUtils::SerializeVec(ranklist);
  MLOGIFR0(MLOG_INFO, "AssignBlocks: ranklist %s", rankstr.c_str());

  return rv;
}
}  // namespace topo
