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

  for (int ts = 0; ts < opts_.num_ts; ts++) {
    MLOGIFR0(MLOG_INFO, "- Running timestep %d...", ts);

    // Create base mesh and print it
    Mesh mesh(dims.x, dims.y, dims.z, lvl);

    // Refine to target leaf count
    int tgt_leafcnt = opts_.tgt_leafcnt;
    int cur_leafcnt = mesh.RefineToTargetLeafcnt(tgt_leafcnt);
    MLOGIFR0(MLOG_INFO, "Refined to %d leaves", cur_leafcnt);

    // Print hierarchy
    PrintUtils::PrintHierarchy(mesh, Mesh::GetRootLoc());

    // Generate ordered mesh and print it
    auto omesh = mesh.GetOrderedMesh();
    if (Globals::my_rank == 0) {
      PrintUtils::PrintOmesh(omesh);
    }

    // int nblocks = omesh.nblocks;
    // std::vector<int> ranklist(nblocks, -1);
    // int rv = AssignBlocks(ranklist, nblocks, opts_.nranks);
    // ABORTIF(rv, "Placement assignment failed!");

    // RunWithOmesh(omesh, ranklist);
  }
}

void MeshDriver::RunWithOmesh(OrderedMesh& omesh, std::vector<int>& ranklist) {
  // Compute load-balanced placement
  int nblocks = omesh.nblocks;
  int rv =
      PlacementUtils::SetupCommMesh(comm_mesh_, omesh, ranklist, opts_.msgsz);
  MLOGIF(rv, MLOG_ERRO, "SetupCommMesh failed!");

  // Allocate boundary variables for communication
  auto s = comm_mesh_.AllocateBvars();
  MLOGIF(s != Status::OK, MLOG_ERRO, "Failed to allocate boundary variables!");
  MLOGIFR0(MLOG_INFO, "Allocated boundary variables");

  for (int rnum = 0; rnum < opts_.num_rounds; rnum++) {
    // Run one communication round
    s = comm_mesh_.DoCommunicationRound();
    MPI_Barrier(MPI_COMM_WORLD);
    MLOGIF(s != Status::OK, MLOG_ERRO, "Communication round failed!");
    MLOGIFR0(MLOG_INFO, "Communication round complete.");
  }

  //   // Print stats and cleanup
  ExtraMetricVec extra_metrics{
      {"policy", opts_.policy},
      {"nblocks", std::to_string(nblocks)},
      {"msgsz", opts_.msgsz.ToString()},
  };

  {
    PrintSectionUtil print_section(MLOG_INFO, "Run Stats", 10);
    comm_mesh_.GenerateStats(extra_metrics, GetLogPath().c_str());
  }
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

  auto coststr = PrintUtils::SerializeVec(costlist, ", ", 10);
  MLOGIFR0(MLOG_INFO, "AssignBlocks: costlist %s", coststr.c_str());

  auto rankstr = PrintUtils::SerializeVec(ranklist, ", ", 10);
  MLOGIFR0(MLOG_INFO, "AssignBlocks: ranklist %s", rankstr.c_str());

  return rv;
}
}  // namespace topo
