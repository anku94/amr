#include "mesh_driver.h"

#include "amr/block.h"
#include "amr/mesh.h"
#include "amr/mesh_utils.h"
#include "amr/print_utils.h"
#include "amr_lb.h"
#include "bench/comm_mesh.h"
#include "bench/metric_utils.h"
#include "bench/newlogger.h"
#include "bench/placement_utils.h"
#include "tools-common/distributions.h"
#include "tools-common/logging.h"

using MeshBlockRef = std::shared_ptr<topo::MeshBlock>;
using DistributionUtils = amr::DistributionUtils;

namespace {
template <typename T>
MPI_Datatype GetMpiType() {
  if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else if (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else {
    static_assert(std::is_same<T, T>::value, "should not reach here");
  }
}

template <typename T>
void BroadcastVec(std::vector<T>& vec, int root) {
  int vecsz = vec.size();
  int rv = 0;

  rv = MPI_Bcast(&vecsz, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_CHECK(rv, "MPI_Bcast failed!");

  if (Globals::my_rank != root) {
    vec.resize(vecsz);
  }

  rv = MPI_Bcast(vec.data(), vecsz, GetMpiType<T>(), root, MPI_COMM_WORLD);
  MPI_CHECK(rv, "MPI_Bcast failed!");
}
}  // namespace

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

    // Generate ordered mesh and print it
    auto omesh = PrepareOmeshSync();
    if (Globals::my_rank == 0) {
      PrintUtils::PrintOmesh(omesh);
    }

    int nblocks = omesh.nblocks;
    std::vector<int> ranklist(nblocks, -1);
    int rv = AssignBlocksSync(ranklist, nblocks, opts_.nranks);
    ABORTIF(rv, "Placement assignment failed!");

    RunWithOmesh(ts, omesh, ranklist);
  }

  {
    PrintSectionUtil print_section(MLOG_INFO, "Run Stats", 10);
    ExtraMetricVec extra_metrics{
        {"policy", opts_.policy},
        {"msgsz-f-e-v", fmtstr(MLOG_BUFSZ, "%d-%d-%d", opts_.msgsz.x,
                               opts_.msgsz.y, opts_.msgsz.z)},
    };

    logger_.AggregateAndWrite(extra_metrics, GetLogPath().c_str());
  }
}

void MeshDriver::RunWithOmesh(int ts, OrderedMesh& omesh,
                              std::vector<int>& ranklist) {
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
    // Barrier to synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);

    // Run one communication round
    logger_.LogRoundBegin();
    s = comm_mesh_.DoCommunicationRound();
    logger_.LogCommEnd();

    // Barrier to synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    logger_.LogBarrierEnd(ts, rnum, nblocks, comm_mesh_.blocks_);

    MLOGIF(s != Status::OK, MLOG_ERRO, "Communication round failed!");
    MLOGIFR0(MLOG_INFO, "Communication round complete.");
  }

  comm_mesh_.ResetBvarsAndBlocks();
}

int MeshDriver::AssignBlocksSync(std::vector<int>& ranklist, int nblocks,
                                 int nranks) {
  int rv = AssignBlocksSingle(ranklist, nblocks, nranks);
  BroadcastVec(ranklist, 0);
  return rv;
}

int MeshDriver::AssignBlocksSingle(std::vector<int>& ranklist, int nblocks,
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

OrderedMesh MeshDriver::PrepareOmeshSync() const {
  // Rank 0 creates a base mesh and broadcasts it to all ranks
  std::vector<int> packed_omesh;

  if (Globals::my_rank == 0) {
    auto omesh = PrepareOmeshSingle();
    omesh.Pack(packed_omesh);
  }

  BroadcastVec(packed_omesh, 0);

  OrderedMesh omesh_recv;
  omesh_recv.Unpack(packed_omesh);
  return omesh_recv;
}

OrderedMesh MeshDriver::PrepareOmeshSingle() const {
  auto dims = opts_.mesh_dims;
  auto lvl = opts_.max_reflvl;

  // Create base mesh
  Mesh mesh(dims.x, dims.y, dims.z, lvl);

  // Refine to target leaf count
  int tgt_leafcnt = opts_.tgt_leafcnt;
  int cur_leafcnt = mesh.RefineToTargetLeafcnt(tgt_leafcnt);
  MLOGIFR0(MLOG_INFO, "Refined to %d leaves", cur_leafcnt);

  // Print hierarchy
  PrintUtils::PrintHierarchy(mesh, Mesh::GetRootLoc());

  auto omesh = mesh.GetOrderedMesh();
  return omesh;
}

}  // namespace topo
