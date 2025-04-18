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

class MetricUtils {
 public:
  // LocStats: local stats for a single rank
  struct LocStats {
    uint64_t totbytes_sent_;
    uint64_t totbytes_rcvd_;
    double totdur_ms_;
  };

  // GlobStats: global stats across all ranks
  struct GlobStats {
    uint64_t totbytes_sent_;
    uint64_t totbytes_rcvd_;
    double totdurms_avg_;
    double totdurms_min_;
    double totdurms_max_;
  };

  struct MetricData {
    std::vector<std::string> header;         // key
    std::vector<std::string> fmtdata_csv;    // fmt for csv
    std::vector<std::string> fmtdata_print;  // fmt for printing

    void AddMetric(std::string key, std::string val_csv,
                   std::string val_print = "") {
      header.push_back(key);
      fmtdata_csv.push_back(val_csv);
      if (!val_print.empty()) {
        fmtdata_print.push_back(val_print);
      } else {
        fmtdata_print.push_back(val_csv);
      }
    }
  };

  static int AggregateStats(LocStats const &local_stats,
                            GlobStats &global_stats) {
    MPI_Reduce(&local_stats.totbytes_sent_, &global_stats.totbytes_sent_, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totbytes_rcvd_, &global_stats.totbytes_rcvd_, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_stats.totdur_ms_, &global_stats.totdurms_avg_, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totdur_ms_, &global_stats.totdurms_min_, 1,
               MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.totdur_ms_, &global_stats.totdurms_max_, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return 0;
  }

  // JoinVec: join a vector of strings with a delimiter
  static std::string JoinVec(const std::vector<std::string> &vec,
                             const std::string &delim) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i != vec.size() - 1) {
        oss << delim;
      }
    }
    return oss.str();
  }

  // LogBytes: log bytes sent/recvd, also compute mbps for both
  static void LogBytes(GlobStats const &gstats, MetricData &md) {
    const uint64_t bytes_per_mb = 1ull << 20;
    char buf[64];

    double mbytes_sent = gstats.totbytes_sent_ * 1.0 / bytes_per_mb;
    double mbytes_recv = gstats.totbytes_rcvd_ * 1.0 / bytes_per_mb;
    double mbps_sent = mbytes_sent / gstats.totdurms_max_;
    double mbps_recv = mbytes_recv / gstats.totdurms_max_;

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
  }

  // LogTime: log durms avg/min/max
  static void LogTime(GlobStats const &gstats, MetricData &md) {
    FMT_AND_ADD("time_avg_ms", gstats.totdurms_avg_, "%.3lf", "%.3lf ms");
    FMT_AND_ADD("time_min_ms", gstats.totdurms_min_, "%.3lf", "%.3lf ms");
    FMT_AND_ADD("time_max_ms", gstats.totdurms_max_, "%.3lf", "%.3lf ms");
  }

  // WriteMetricData: write the metric data to a file
  // - If the file does not exist, create it and write the header
  // - Append the data to the file
  static void WriteMetricData(const char *file, MetricData const &md) {
    FILE *f = fopen(file, "a+");
    if (f == nullptr) return;

    if (!FileExists(file)) {
      std::string header_str = JoinVec(md.header, ",");
      fprintf(f, "%s\n", header_str.c_str());
    }

    std::string data_str = JoinVec(md.fmtdata_csv, ",");
    fprintf(f, "%s\n", data_str.c_str());
  }

  static bool FileExists(const char *file) {
    struct stat statbuf;
    return stat(file, &statbuf) == 0;
  }
};

namespace topo {
void Logger::DrainBlockData(std::vector<MeshBlockRef> &blocks) {
  totbytes_sent_ = 0;
  totbytes_rcvd_ = 0;

  for (auto b : blocks) {
    totbytes_sent_ += b->BytesSent();
    totbytes_rcvd_ += b->BytesRcvd();
  }

  auto delta = end_us_ - start_us_;
  totdur_ms_ += (delta / 1e3);  // us-to-ms
}

void Logger::AggregateAndWrite(ExtraMetricVec &extra_metrics) {
  MetricUtils::LocStats locstats{
      .totbytes_sent_ = totbytes_sent_,
      .totbytes_rcvd_ = totbytes_rcvd_,
      .totdur_ms_ = totdur_ms_,
  };
  MetricUtils::GlobStats gstats;

  // Aggregate global stats
  int rv = MetricUtils::AggregateStats(locstats, gstats);
  ABORTIF(rv, "MPI_Reduce failed!");
  if (Globals::my_rank != 0) return;

  const int nranks = GetNumRanks();
  gstats.totdurms_avg_ /= nranks;

  MetricUtils::MetricData md;

  // Add extra metrics first
  for (const auto &em : extra_metrics) {
    md.AddMetric(em.first, em.second);
  }

  md.AddMetric("nranks", std::to_string(nranks));
  md.AddMetric("meshgen_method", MeshGenMethodToStrUtil());
  md.AddMetric("nrounds", std::to_string(num_obs_));

  MetricUtils::LogBytes(gstats, md);
  MetricUtils::LogTime(gstats, md);

  // Write to log file
  const char *log_fpath = "/tmp/bench_log.csv";
  MLOGIFR0(MLOG_INFO, "Adding run stats to log file: %s", log_fpath);
  MetricUtils::WriteMetricData(log_fpath, md);

  // Also print to console
  std::string sep(10, '-');
  MLOGIFR0(MLOG_INFO, "%s Run stats %s", sep.c_str(), sep.c_str());
  for (int midx = 0; midx < md.fmtdata_csv.size(); ++midx) {
    MLOGIFR0(MLOG_INFO, "%15s: %s", md.header[midx].c_str(),
             md.fmtdata_print[midx].c_str());
  }
  MLOGIFR0(MLOG_INFO, "%s-----------%s", sep.c_str(), sep.c_str());
}

int Logger::GetNumRanks() const {
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  return num_ranks;
}
}  // namespace topo
