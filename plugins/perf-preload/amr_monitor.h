#pragma once

#include "common.h"
#include "detailed_logger.h"
#include "logging.h"
#include "metric.h"
#include "metric_util.h"
#include "p2p.h"
#include "print_utils.h"
#include "types.h"

#include <cinttypes>
#include <glog/logging.h>
#include <pdlfs-common/env.h>

namespace amr {

class AMRMonitor {
 public:
  AMRMonitor(pdlfs::Env* env, int rank, int nranks)
      : env_(env),
        rank_(rank),
        nranks_(nranks),
        tswise_logger_(
            AMROptUtils::GetTswiseOutputFile(amr_opts, env_, rank_)) {
    google::InitGoogleLogging("amrmon");

    if (rank == 0) {
      logv(__LOG_ARGS__, LOG_INFO, "AMRMonitor initializing.");
      AMROptUtils::LogOpts(amr_opts);

      if (amr_opts.tswise_enabled) {
        logv(__LOG_ARGS__, LOG_WARN,
             "Timestep-wise logging is enabled! This may produce lots of "
             "data!!");
      }
    }
  }

  ~AMRMonitor() {
    LogMetrics();
    logvat0(__LOG_ARGS__, LOG_INFO, "AMRMonitor destroyed on rank %d", rank_);
  }

  uint64_t Now() const {
    auto now = env_->NowMicros();
    return now;
  }

  void LogMPICollective(const char* type, uint64_t time_us) {
    LogKey(times_us_, type, time_us);
  }

  void LogStackBegin(const char* type, const char* key) {
    if (stack_map_.find(type) == stack_map_.end()) {
      stack_map_[type] = ProfStack();
    }

    auto& s = stack_map_[type];
    s.push(StackOpenPair(key, Now()));
  }

  void LogStackEnd(const char* type) {
    if (stack_map_.find(type) == stack_map_.end()) {
      logv(__LOG_ARGS__, LOG_WARN, "type %s not found in stack_map_.", type);
      return;
    }

    auto& s = stack_map_[type];
    if (s.empty()) {
      logv(__LOG_ARGS__, LOG_WARN, "stack %s is empty.", type);
      return;
    }

    auto end_time = Now();
    auto begin_time = s.top().second;
    auto elapsed_time = end_time - begin_time;

    auto key = s.top().first;
    LogKey(times_us_, key.c_str(), elapsed_time);

    s.pop();
  }

  inline int GetMPITypeSizeCached(MPI_Datatype datatype) {
    if (mpi_datatype_sizes_.find(datatype) == mpi_datatype_sizes_.end()) {
      int size;
      int rv = PMPI_Type_size(datatype, &size);
      if (rv != MPI_SUCCESS) {
        logv(__LOG_ARGS__, LOG_WARN, "PMPI_Type_size failed");
      }

      mpi_datatype_sizes_[datatype] = size;
    }

    return mpi_datatype_sizes_[datatype];
  }

  void LogMPISend(int dest_rank, MPI_Datatype datatype, int count) {
    auto size = count * GetMPITypeSizeCached(datatype);
    p2p_comm_.LogSend(dest_rank, size);
  }

  void LogMPIRecv(int src_rank, MPI_Datatype datatype, int count) {
    auto size = count * GetMPITypeSizeCached(datatype);
    p2p_comm_.LogRecv(src_rank, size);
  }

  void LogKey(MetricMap& map, const char* key, uint64_t val) {
    logvat0(__LOG_ARGS__, LOG_DBG2, "Rank %d: key %s, val: %" PRIu64 "\n",
            rank_, key, val);
    // must use iterators because Metric class has const variables,
    // and therefore can not be assigned to and all
    auto it = map.find(key);
    if (it == map.end()) {
      map.insert({key, Metric(key, rank_)});
      it = map.find(key);
    }

    it->second.LogInstance(val);

    tswise_logger_.LogKey(key, val);
  }

  StringVec GetCommonMetrics() {
    logvat0(__LOG_ARGS__, LOG_DBUG, "Entering GetCommonMetrics.");

    StringVec local_metrics;
    for (auto& kv : times_us_) {
      local_metrics.push_back(kv.first);
    }

    auto intersection_computer = CommonComputer(local_metrics, rank_, nranks_);
    auto intersection = intersection_computer.Compute();

    logvat0(__LOG_ARGS__, LOG_DBUG, "Exiting GetCommonMetrics.");
    return intersection;
  }

  void LogMetrics() {
    logvat0(__LOG_ARGS__, LOG_DBUG, "Entering LogMetrics.");

    // First, need to get metrics that are logged on all ranks
    // as collectives will block on ranks that are missing a given metric
    StringVec common_metrics = GetCommonMetrics();

    auto compute_metric_stats =
        Metric::CollectMetrics(common_metrics, times_us_);
    auto compute_metric_str = MetricPrintUtils::SortAndSerialize(
        compute_metric_stats, amr_opts.print_topk);

    if (rank_ == 0) {
      fprintf(stderr, "%s\n\n", compute_metric_str.c_str());
    }

    if (amr_opts.rankwise_enabled) {
      CollectMetricsDetailed(common_metrics);
    }

    if (amr_opts.p2p_enable_matrix_put) {
      logvat0(__LOG_ARGS__, LOG_DBUG, "Collecting P2P matrix with RMA PUT.");

      auto p2p_matrix_str = p2p_comm_.CollectAndAnalyze(rank_, nranks_, true);
      if (rank_ == 0) {
        fprintf(stderr, "%s\n", p2p_matrix_str.c_str());
      }
    }

    if (amr_opts.p2p_enable_matrix_reduce) {
      logvat0(__LOG_ARGS__, LOG_DBUG, "Collecting P2P matrix with RMA PUT.");

      auto p2p_matrix_str = p2p_comm_.CollectAndAnalyze(rank_, nranks_, false);
      if (rank_ == 0) {
        fprintf(stderr, "%s\n", p2p_matrix_str.c_str());
      }
    }

    logvat0(__LOG_ARGS__, LOG_DBUG, "Exiting LogMetrics.");
  }

  void CollectMetricsDetailed(StringVec const& metrics) {
    pdlfs::WritableFile* f;
    pdlfs::Status s =
        env_->NewWritableFile(amr_opts.rankwise_fpath.c_str(), &f);
    if (!s.ok()) {
      logv(__LOG_ARGS__, LOG_WARN, "Failed to open file %s",
           amr_opts.rankwise_fpath.c_str());
      return;
    }

    for (auto& m : metrics) {
      auto it = times_us_.find(m);
      if (it != times_us_.end()) {
        auto metric_str = it->second.GetMetricRankwise(nranks_);
        if (rank_ == 0) {
          f->Append(pdlfs::Slice(metric_str.c_str(), metric_str.size()));
        }
      }
    }
  }

 private:
  MetricMap times_us_;
  StackMap stack_map_;

  P2PCommCollector p2p_comm_;

  std::unordered_map<MPI_Datatype, uint32_t> mpi_datatype_sizes_;

  std::unordered_map<std::string, uint64_t> begin_times_us_;

  pdlfs::Env* const env_;
  const int rank_;
  const int nranks_;
  TimestepwiseLogger tswise_logger_;
};
}  // namespace amr
