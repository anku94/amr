//
// Created by Ankush J on 4/12/22.
//

#include "logger.h"

#include <inttypes.h>
#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "amr/block.h"
#include "metric_utils.h"

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
  // switch (topo::Globals::driver_opts.meshgen_method) {
  // case MeshGenMethod::Ring:
  //   return "Ring";
  //   break;
  // case MeshGenMethod::AllToAll:
  //   return "AllToALl";
  //   break;
  // case MeshGenMethod::FromSingleTSTrace:
  //   return std::string("SingleTS:") + topo::Globals::driver_opts.trace_root;
  //   break;
  // case MeshGenMethod::FromMultiTSTrace:
  //   return std::string("MultiTS:") + topo::Globals::driver_opts.trace_root;
  //   break;
  // default:
  //   break;
  // }

  return "UNKNOWN";
}
}  // namespace

class LoggerUtils {
 public:
  // LocStats: local stats for a single rank
  struct LocStats {
    uint64_t totbytes_sent_;
    uint64_t totbytes_rcvd_;
    uint64_t totcnt_sent_;
    uint64_t totcnt_rcvd_;
    double totdur_ms_;
  };

  // GlobStats: global stats across all ranks
  struct GlobStats {
    uint64_t totbytes_sent_;
    uint64_t totbytes_rcvd_;
    uint64_t totcnt_sent_;
    uint64_t totcnt_rcvd_;
    double totdurms_avg_;
    double totdurms_min_;
    double totdurms_max_;
  };

  static int AggregateStats(LocStats const &local_stats,
                            GlobStats &global_stats) {
    MPI_Reduce(&local_stats.totbytes_sent_, &global_stats.totbytes_sent_, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totbytes_rcvd_, &global_stats.totbytes_rcvd_, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totcnt_sent_, &global_stats.totcnt_sent_, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totcnt_rcvd_, &global_stats.totcnt_rcvd_, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_stats.totdur_ms_, &global_stats.totdurms_avg_, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totdur_ms_, &global_stats.totdurms_min_, 1,
               MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totdur_ms_, &global_stats.totdurms_max_, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return 0;
  }

  // LogComm: log bytes sent/recvd, also compute mbps for both
  static void LogComm(GlobStats const &gstats, topo::MetricUtils::MetricData &md) {
    const uint64_t bytes_per_mb = 1ull << 20;
    double dursec = gstats.totdurms_max_ * 1.0 / 1000.0;
    ABORTIF(dursec <= 0.0, "Duration is <= 0!");

    double mbytes_sent = gstats.totbytes_sent_ * 1.0 / bytes_per_mb;
    double mbytes_recv = gstats.totbytes_rcvd_ * 1.0 / bytes_per_mb;
    double mbps_sent = mbytes_sent / dursec;
    double mbps_recv = mbytes_recv / dursec;

#define FMT_AND_ADD(k, v, vfmtcsv, vfmtprint)             \
  {                                                       \
    char vbufcsv[1024];                                   \
    char vbufprint[1024];                                 \
    snprintf(vbufcsv, sizeof(vbufcsv), vfmtcsv, v);       \
    snprintf(vbufprint, sizeof(vbufprint), vfmtprint, v); \
    md.AddMetric(k, vbufcsv, vbufprint);                  \
  }

    FMT_AND_ADD("mbytes_sent", mbytes_sent, "%.2lf", "%.2lf MB");
    FMT_AND_ADD("mbytes_recv", mbytes_recv, "%.2lf", "%.2lf MB");
    FMT_AND_ADD("mbps_sent", mbps_sent, "%.4lf", "%.4lf MB/s");
    FMT_AND_ADD("mbps_recv", mbps_recv, "%.4lf", "%.4lf MB/s");
    FMT_AND_ADD("cnt_sent", gstats.totcnt_sent_, "%" PRIu64, "%" PRIu64);
    FMT_AND_ADD("cnt_recv", gstats.totcnt_rcvd_, "%" PRIu64, "%" PRIu64);
  }

  // LogTime: log durms avg/min/max
  static void LogTime(GlobStats const &gstats, topo::MetricUtils::MetricData &md) {
    FMT_AND_ADD("time_avg_ms", gstats.totdurms_avg_, "%.3lf", "%.3lf ms");
    FMT_AND_ADD("time_min_ms", gstats.totdurms_min_, "%.3lf", "%.3lf ms");
    FMT_AND_ADD("time_max_ms", gstats.totdurms_max_, "%.3lf", "%.3lf ms");
  }
};

namespace topo {
void Logger::DrainBlockData(std::vector<MeshBlockRef> &blocks) {
  totbytes_sent_ = 0;
  totbytes_rcvd_ = 0;

  for (auto b : blocks) {
    totbytes_sent_ += b->BytesSent();
    totbytes_rcvd_ += b->BytesRcvd();
    totcnt_sent_ += b->CountSent();
    totcnt_rcvd_ += b->CountRcvd();
  }

  auto delta = end_us_ - start_us_;
  totdur_ms_ += (delta / 1e3);  // us-to-ms
}

void Logger::AggregateAndWrite(ExtraMetricVec &extra_metrics,
                                const char *log_fpath) {
  LoggerUtils::LocStats locstats{
      .totbytes_sent_ = totbytes_sent_,
      .totbytes_rcvd_ = totbytes_rcvd_,
      .totcnt_sent_ = totcnt_sent_,
      .totcnt_rcvd_ = totcnt_rcvd_,
      .totdur_ms_ = totdur_ms_,
  };
  LoggerUtils::GlobStats gstats;

  // Aggregate global stats
  int rv = LoggerUtils::AggregateStats(locstats, gstats);
  ABORTIF(rv, "MPI_Reduce failed!");
  if (Globals::my_rank != 0) return;

  const int nranks = GetNumRanks();
  gstats.totdurms_avg_ /= nranks;

  topo::MetricUtils::MetricData md;

  // Add extra metrics first
  for (const auto &em : extra_metrics) {
    md.AddMetric(em.first, em.second);
  }

  md.AddMetric("nranks", std::to_string(nranks));
  md.AddMetric("meshgen_method", MeshGenMethodToStrUtil());
  md.AddMetric("nrounds", std::to_string(num_obs_));

  LoggerUtils::LogComm(gstats, md);
  LoggerUtils::LogTime(gstats, md);

  // Write to log file
  MLOGIFR0(MLOG_INFO, "Adding run stats to log file: %s", log_fpath);
  MetricUtils::WriteMetricData(log_fpath, md);

  // Also print to console
  for (int midx = 0; midx < md.fmtdata_csv.size(); ++midx) {
    MLOGIFR0(MLOG_INFO, "%15s: %s", md.header[midx].c_str(),
             md.fmtdata_print[midx].c_str());
  }
}

int Logger::GetNumRanks() const {
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  return num_ranks;
}
}  // namespace topoBytes
