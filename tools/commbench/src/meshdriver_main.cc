#include <mpi.h>

#include <cstdio>

#include "amr/mesh_utils.h"
#include "bench/mesh_driver.h"
#include "tools-common/config_parser.h"
#include "tools-common/logging.h"

// using namespace topo;
// using namespace topo;

// Set AMRLB_CONFIG to the config
topo::MeshDriverOpts GetOpts() {
  auto dims_xyz = amr::ConfigUtils::GetParamOrDefault<int>("dims_xyz", 2);
  auto max_reflvl = amr::ConfigUtils::GetParamOrDefault<int>("max_reflvl", 4);
  auto tgt_leafcnt =
      amr::ConfigUtils::GetParamOrDefault<int>("target_leafcnt", 15);

  auto f_msgsz =
      amr::ConfigUtils::GetParamOrDefault<int>("f_msgsz_bytes", 1024);
  auto e_msgsz =
      amr::ConfigUtils::GetParamOrDefault<int>("e_msgsz_bytes", 1024);
  auto v_msgsz =
      amr::ConfigUtils::GetParamOrDefault<int>("v_msgsz_bytes", 1024);
  auto msgsz = topo::Vec3i(f_msgsz, e_msgsz, v_msgsz);

  auto jobdir = amr::ConfigUtils::GetParamOrDefault("jobdir", "/tmp");
  auto log_fname =
      amr::ConfigUtils::GetParamOrDefault("log_fname", "bench.log");

  auto policy = amr::ConfigUtils::GetParamOrDefault("policy", "baseline");
  auto num_ts = amr::ConfigUtils::GetParamOrDefault<int>("num_ts", 1);
  auto num_rounds = amr::ConfigUtils::GetParamOrDefault<int>("num_rounds", 1);

  {
    topo::PrintSectionUtil print_section(MLOG_DBG0, "Parsed Config", 10);
    MLOG(MLOG_DBG0, "%15s: %d", "Dims xyz", dims_xyz);
    MLOG(MLOG_DBG0, "%15s: %d", "Max reflvl", max_reflvl);
    MLOG(MLOG_DBG0, "%15s: %d", "Target leafcnt", tgt_leafcnt);
    MLOG(MLOG_DBG0, "%15s: %s", "Msg sizes (bytes)", msgsz.ToString().c_str());
    MLOG(MLOG_DBG0, "%15s: %s", "Job dir", jobdir);
    MLOG(MLOG_DBG0, "%15s: %s", "Log fname", log_fname);
    MLOG(MLOG_DBG0, "%15s: %s", "Policy", policy);
    MLOG(MLOG_DBG0, "%15s: %d", "Num ts", num_ts);
    MLOG(MLOG_DBG0, "%15s: %d", "Num rounds", num_rounds);
  }

  topo::MeshDriverOpts opts;
  opts.mesh_dims = topo::Vec3i(dims_xyz, dims_xyz, dims_xyz);
  opts.max_reflvl = max_reflvl;
  opts.tgt_leafcnt = tgt_leafcnt;
  opts.msgsz = msgsz;
  opts.policy = policy;
  opts.jobdir = jobdir;
  opts.log_fname = log_fname;
  opts.num_ts = num_ts;
  opts.num_rounds = num_rounds;

  return opts;
}

void AdjustLogging(int my_rank) {
  FLAGS_log_prefix = false;
  FLAGS_logtostderr = (my_rank == 0) ? true : false;

  // AMRLB_LOGDIR is the env var for log dir
  auto logdir = std::getenv("AMRLB_LOGDIR");
  if (!logdir) return;

  // Write to logdir/rank<r>.log
  if (my_rank != 0) {
    char log_fpath[1024];
    snprintf(log_fpath, sizeof(log_fpath), "%s/rank%d.log", logdir, my_rank);
    google::SetLogDestination(google::GLOG_INFO, log_fpath);
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  MPI_Init(&argc, &argv);

  int my_rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // Adjust logging before getting opts to log opts
  AdjustLogging(my_rank);
  auto opts = GetOpts();
  opts.my_rank = my_rank;
  opts.nranks = nranks;

  topo::MeshDriver driver(opts);
  driver.Run();

  MPI_Finalize();
  return 0;
}
