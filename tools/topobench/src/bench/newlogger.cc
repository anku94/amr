#include "newlogger.h"

using topo::ExtraMetricVec;
using topo::MetricUtils;

class Utils {
 public:
  struct AggRoundStats {
    int ts;                     // timestep
    int round;                  // round within timestep
    int nblocks;
    uint64_t totbytes_sent;     // total bytes sent
    uint64_t totbytes_rcvd;     // total bytes received
    uint64_t totcnt_sent;       // total count sent
    uint64_t totcnt_rcvd;       // total count received
    double totdurms_msgcomm_avg;  // dur of msg comm
    double totdurms_msgcomm_min;  // min dur of msg comm
    double totdurms_msgcomm_max;  // max dur of msg comm
    double totdurms_mpi_bar;       // dur at rank0 after bar
  };

  static int AggregateStats(int nranks, topo::RoundStats const &ls,
                            AggRoundStats &gs) {
    gs.ts = ls.ts;
    gs.round = ls.round;
    gs.totdurms_mpi_bar = ls.totdurms_mpi_bar;
    gs.nblocks = ls.nblocks;

    MPI_Reduce(&ls.totbytes_sent, &gs.totbytes_sent, 1, MPI_UINT64_T, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.totbytes_rcvd, &gs.totbytes_rcvd, 1, MPI_UINT64_T, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.totcnt_sent, &gs.totcnt_sent, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&ls.totcnt_rcvd, &gs.totcnt_rcvd, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&ls.totdurms_msg_comm, &gs.totdurms_msgcomm_avg, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.totdurms_msg_comm, &gs.totdurms_msgcomm_min, 1, MPI_DOUBLE,
               MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.totdurms_msg_comm, &gs.totdurms_msgcomm_max, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    gs.totdurms_msgcomm_avg /= nranks;

    return 0;
  }

  static int AggregateStatsVec(int nranks,
                               std::vector<topo::RoundStats> const &ls_vec,
                               std::vector<AggRoundStats> &gs_vec) {
    for (auto &ls : ls_vec) {
      AggRoundStats gs;
      int rv = AggregateStats(nranks, ls, gs);
      ABORTIF(rv, "Failed to aggregate stats!");
      gs_vec.push_back(gs);
    }

    return 0;
  }

  static void AddExtraMetrics(topo::MetricUtils::MetricData &md,
                              ExtraMetricVec const &extra_metrics) {
    for (auto &em : extra_metrics) {
      md.AddMetric(em.first, em.second);
    }
  }

  static std::vector<topo::MetricUtils::MetricData> GetMetricDataVec(
      std::vector<AggRoundStats> const &gs_vec,
      ExtraMetricVec const &extra_metrics) {
    std::vector<topo::MetricUtils::MetricData> md_vec;
    for (auto &gs : gs_vec) {
      topo::MetricUtils::MetricData md;
      AddExtraMetrics(md, extra_metrics);

      md.AddMetric("ts", std::to_string(gs.ts));
      md.AddMetric("round", std::to_string(gs.round));
      md.AddMetric("nblocks", std::to_string(gs.nblocks));
      md.AddMetric("totbytes_sent", std::to_string(gs.totbytes_sent));
      md.AddMetric("totbytes_rcvd", std::to_string(gs.totbytes_rcvd));
      md.AddMetric("totcnt_sent", std::to_string(gs.totcnt_sent));
      md.AddMetric("totcnt_rcvd", std::to_string(gs.totcnt_rcvd));
      md.AddMetric("totdurms_msgcomm_avg", std::to_string(gs.totdurms_msgcomm_avg));
      md.AddMetric("totdurms_msgcomm_min", std::to_string(gs.totdurms_msgcomm_min));
      md.AddMetric("totdurms_msgcomm_max", std::to_string(gs.totdurms_msgcomm_max));
      md.AddMetric("totdurms_mpi_bar", std::to_string(gs.totdurms_mpi_bar));
      md_vec.push_back(md);
    }

    return md_vec;
  }

  static void LogRunSummary(std::vector<AggRoundStats> const &gs_vec) {
#define LOG_KVSTAT(key, val) \
  MLOGIFR0(MLOG_INFO, "  %20s = %s", key, val.c_str());

    double totbytes_sent = 0;
    double totdurms_mpi_bar = 0;
    for (auto &gs : gs_vec) {
      totbytes_sent += gs.totbytes_sent;
      totdurms_mpi_bar += gs.totdurms_mpi_bar;
    }

    int max_ts = gs_vec.back().ts;
    int max_round = gs_vec.back().round;


    LOG_KVSTAT("Rounds (ts * rperts)", fmtstr(64, "%zu", gs_vec.size()));
    LOG_KVSTAT("Num timesteps", fmtstr(64, "%d", max_ts + 1));
    LOG_KVSTAT("Num rounds", fmtstr(64, "%d", max_round + 1));
    LOG_KVSTAT("Total data sent",
               fmtstr(64, "%.2f MB", totbytes_sent / (1 << 20)));
    LOG_KVSTAT("Total round time", fmtstr(64, "%.2f ms", totdurms_mpi_bar));
  }
};

namespace topo {
void NewLogger::AggregateAndWrite(ExtraMetricVec &extra_metrics,
                                  const char *log_fpath) {
  int nranks = Globals::nranks;
  std::vector<Utils::AggRoundStats> gs_vec;

  // Aggregate stats using MPI collectives
  int rv = Utils::AggregateStatsVec(nranks, round_stats_, gs_vec);
  ABORTIF(rv, "Failed to aggregate stats!");

  // Serialize stats into strings
  std::vector<topo::MetricUtils::MetricData> md_vec =
      Utils::GetMetricDataVec(gs_vec, extra_metrics);

  if (Globals::my_rank != 0) {
    // Non-master ranks do not write to file
    return;
  }

  // Write stats to file
  for (auto &mdobj : md_vec) {
    MetricUtils::WriteMetricData(log_fpath, mdobj);
  }

  Utils::LogRunSummary(gs_vec);
}
}  // namespace topo
