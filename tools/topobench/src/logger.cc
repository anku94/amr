//
// Created by Ankush J on 4/12/22.
//

#include "logger.h"

#include "block.h"
#include "globals.h"

#include <inttypes.h>
#include <mpi.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {
std::string GetMPIStr() {
  char version_str[MPI_MAX_LIBRARY_VERSION_STRING];
  int vstrlen;
  MPI_Get_library_version(version_str, &vstrlen);
  std::string delim = " ";
  std::string s = version_str;
  std::string token = s.substr(0, s.find(delim));
  return token;
}

const std::string MeshGenMethodToStrUtil() {
  switch (Globals::driver_opts.meshgen_method) {
  case MeshGenMethod::Ring:
    return "Ring";
    break;
  case MeshGenMethod::AllToAll:
    return "AllToALl";
    break;
  case MeshGenMethod::FromSingleTSTrace:
    return std::string("SingleTS:") + Globals::driver_opts.trace_root;
    break;
  case MeshGenMethod::FromMultiTSTrace:
    return std::string("MultiTS:") + Globals::driver_opts.trace_root;
    break;
  default:
    break;
  }

  return "UNKNOWN";
}
} // namespace

void Logger::LogData(std::vector<std::shared_ptr<MeshBlock>> &blocks_) {
  total_sent_ = 0;
  total_rcvd_ = 0;

  for (auto b : blocks_) {
    total_sent_ += b->BytesSent();
    total_rcvd_ += b->BytesRcvd();
  }

  auto delta = end_ms_ - start_ms_;
  total_time_ += (delta / 1000.0); // ms-to-s
}

void Logger::Aggregate() {
  uint64_t global_sent, global_rcvd;

  double global_time_avg, global_time_min, global_time_max;

  MPI_Reduce(&total_sent_, &global_sent, 1, MPI_UINT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_rcvd_, &global_rcvd, 1, MPI_UINT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);

  MPI_Reduce(&total_time_, &global_time_avg, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_time_, &global_time_min, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_time_, &global_time_max, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (Globals::my_rank != 0)
    return;

  const int nranks = GetNumRanks();
  global_time_avg /= nranks;

  const uint64_t bytes_per_mb = 1ull << 30;
  double global_sent_mb = global_sent * 1.0 / bytes_per_mb;
  double global_rcvd_mb = global_rcvd * 1.0 / bytes_per_mb;

  double sent_mbps = global_sent_mb / global_time_avg;
  double rcvd_mbps = global_rcvd_mb / global_time_avg;
  logv(__LOG_ARGS__, LOG_INFO, "Bytes Exchanged: %" PRIu64 " B/%" PRIu64 " B",
       global_sent, global_rcvd);
  logv(__LOG_ARGS__, LOG_INFO, "Bytes Exchanged: %.2lf MB/%.2lf MB",
       global_sent_mb, global_rcvd_mb);
  logv(__LOG_ARGS__, LOG_INFO,
       "Effective b/w SEND: %.4lf MB/s RECV: %.4lf MB/s", sent_mbps, rcvd_mbps);
  logv(__LOG_ARGS__, LOG_INFO,
       "Time Avg: %.2lf ms, Min: %.2lf ms, Max: %.2lf ms (%d rounds)",
       global_time_avg * 1e3, global_time_min * 1e3, global_time_max * 1e3,
       num_obs_);

  LogRun(global_sent_mb, sent_mbps, global_rcvd_mb, rcvd_mbps,
         global_time_avg * 1e3, global_time_min * 1e3, global_time_max * 1e3,
         num_obs_);
}

void Logger::LogRun(double send_mb, double send_mbps, double recv_mb,
                    double recv_mbps, double time_avg_ms, double time_min_ms,
                    double time_max_ms, int num_obs) {
  struct stat statbuf;

  auto log_fpath = std::string(Globals::driver_opts.job_dir) + "/bench_log.csv";

  if (stat(log_fpath.c_str(), &statbuf) != 0) {
    FILE *f = fopen(log_fpath.c_str(), "w");
    if (f == nullptr)
      return;

    fprintf(f, "mpi_prov,send_mb,send_mbps,recv_mb,recv_mbps,"
               "time_avg_ms,time_min_ms,time_max_ms,num_obs,meshgen_method\n");
    fclose(f);
  }

  FILE *f = fopen(log_fpath.c_str(), "a+");
  if (f == nullptr)
    return;

  const std::string mpi_str = ::GetMPIStr();
  const std::string topo_str = ::MeshGenMethodToStrUtil();

  fprintf(f,
          "%s,%.6lf,%.6lf,%.6lf,%.6lf," // send-recv mb/mbps
          "%.3lf,%.3lf,%.3lf,%d,"       // time avg-min-max, num_obs
          "%s\n",                       // time avg-min-max
          mpi_str.c_str(), send_mb, send_mbps, recv_mb, recv_mbps, time_avg_ms,
          time_min_ms, time_max_ms, num_obs, topo_str.c_str());

  fclose(f);

  return;
}

int Logger::GetNumRanks() const {
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  return num_ranks;
}
