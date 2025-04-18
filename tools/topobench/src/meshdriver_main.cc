#include <mpi.h>

#include <cstdio>

#include "amr/mesh_utils.h"
#include "bench/mesh_driver.h"
#include "config_parser.h"
#include "globals.h"
#include "logging.h"

// using namespace topo;
// using namespace topo;

// Set AMRLB_CONFIG to the config
topo::MeshDriverOpts GetOpts() {
  auto jobdir = amr::ConfigUtils::GetParamOrDefault("jobdir", "/tmp");
  auto dims_xyz = amr::ConfigUtils::GetParamOrDefault<int>("dims_xyz", 2);
  auto max_reflvl = amr::ConfigUtils::GetParamOrDefault<int>("max_reflvl", 4);
  auto policy = amr::ConfigUtils::GetParamOrDefault("policy", "baseline");

  MLOG(MLOG_INFO, "Job dir: %s", jobdir);
  MLOG(MLOG_INFO, "Dims xyz: %d", dims_xyz);
  MLOG(MLOG_INFO, "Max reflvl: %d", max_reflvl);
  MLOG(MLOG_INFO, "Policy: %s", policy);

  topo::MeshDriverOpts opts;
  opts.mesh_dims = topo::Vec3i(dims_xyz, dims_xyz, dims_xyz);
  opts.max_reflvl = max_reflvl;
  opts.policy = policy;
  opts.jobdir = jobdir;

  return opts;
}

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  FLAGS_log_prefix = false;
  google::InitGoogleLogging(argv[0]);

  auto opts = GetOpts();

  MPI_Init(&argc, &argv);

  int my_rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  opts.my_rank = my_rank;
  opts.nranks = nranks;

  topo::MeshDriver driver(opts);
  driver.Run();

  MPI_Finalize();
  return 0;
}
