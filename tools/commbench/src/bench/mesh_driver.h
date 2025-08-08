#pragma once

#include <glog/logging.h>
#include <mpi.h>

#include <vector>

#include "amr/mesh_utils.h"
#include "bench/comm_mesh.h"
#include "newlogger.h"
#include "tools-common/logging.h"

namespace topo {
struct MeshDriverOpts {
  topo::Vec3i mesh_dims;  // mesh dims as blocks in x,y,z
  int max_reflvl;         // max refinement level
  int tgt_leafcnt;        // target leaf count
  topo::Vec3i msgsz;      // face/edge/vertex msg sizes
  int my_rank;            // local MPI rank
  int nranks;             // total MPI ranks
  std::string policy;     // placement policy
  std::string jobdir;     // job directory
  std::string log_fname;  // log file name
  int num_ts;             // num timesteps
  int num_rounds;         // num rounds/timestep
};

class PrintSectionUtil {
 public:
  PrintSectionUtil(int loglvl, const std::string& section_name, int sep_len)
      : loglvl_(loglvl),
        name_len_(section_name.length()),
        sep_(std::string(sep_len, '-')) {
    MLOG(loglvl_, "%s%s%s", sep_.c_str(), section_name.c_str(), sep_.c_str());
  }

  ~PrintSectionUtil() {
    MLOG(loglvl_, "%s%s%s", sep_.c_str(), std::string(name_len_, '-').c_str(),
         sep_.c_str());
  }

 private:
  int loglvl_;
  int name_len_;
  std::string sep_;
};

class MeshDriver {
 public:
  MeshDriver(const MeshDriverOpts& opts);

  void PrintOpts() {
    PrintSectionUtil print_section(MLOG_INFO, "Meshdriver Opts", 10);

    MLOG(MLOG_INFO, "%15s: %s", "Mesh dims",
         opts_.mesh_dims.ToString().c_str());
    MLOG(MLOG_INFO, "%15s: %d", "Max reflvl", opts_.max_reflvl);
    MLOG(MLOG_INFO, "%15s: %d", "Target leafcnt", opts_.tgt_leafcnt);
    MLOG(MLOG_INFO, "%15s: %s", "Msg sizes", opts_.msgsz.ToString().c_str());
    MLOG(MLOG_INFO, "%15s: %s", "Policy", opts_.policy.c_str());
    MLOG(MLOG_INFO, "%15s: %d", "Num ts", opts_.num_ts);
    MLOG(MLOG_INFO, "%15s: %d", "Num rounds", opts_.num_rounds);
    MLOG(MLOG_INFO, "%15s: %d", "Total ranks", opts_.nranks);

    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MLOG(MLOG_INFO, "%15s: %d", "MPI ranks", nranks);
    ABORTIF(nranks != opts_.nranks, "MPI ranks != nranks in opts");
  }

  // Run: Run for num_ts timesteps
  void Run();

  // RunWithOmesh: Run with an ordered mesh for nrounds
  // (with a single mesh)
  void RunWithOmesh(int ts, OrderedMesh& omesh, std::vector<int>& ranklist);

 private:
  const MeshDriverOpts opts_;
  CommMesh comm_mesh_;
  NewLogger logger_;

  std::string GetLogPath() const {
    return opts_.jobdir + "/" + opts_.log_fname;
  }

  // AssignBlocksSync: broadcasting wrapper around AssignBlocksSingle
  int AssignBlocksSync(std::vector<int>& ranklist, int nblocks, int nranks);

  // AssignBlocksSingle: populate ranklist using a synthetic costlist
  // generated using distribution, + placement scheme in opts.policy
  int AssignBlocksSingle(std::vector<int>& ranklist, int nblocks, int nranks);

  // PrepareOmeshSync: broadcast wrapper around PrepareOmeshSingle
  OrderedMesh PrepareOmeshSync() const;

  // PrepareOmeshSingle: create a single mesh and refine to tgt_leafcnt
  OrderedMesh PrepareOmeshSingle() const;
};
}  // namespace topo
