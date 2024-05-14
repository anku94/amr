#pragma once

#include "common.h"
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
      : env_(env), rank_(rank), nranks_(nranks) {
    google::InitGoogleLogging("amrmon");

    if (rank == 0) {
      Info(__LOG_ARGS__, "AMRMonitor initializing.");
      AMROptUtils::LogOpts(amr_opts);
    }
  }

  ~AMRMonitor() {
    LogMetrics();
    Verbose(__LOG_ARGS__, 1, "AMRMonitor destroyed on rank %d", rank_);
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
      Warn(__LOG_ARGS__, "type %s not found in stack_map_.", type);
      return;
    }

    auto& s = stack_map_[type];
    if (s.empty()) {
      Warn(__LOG_ARGS__, "stack %s is empty.", type);
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
        Warn(__LOG_ARGS__, "PMPI_Type_size failed");
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
    Verbose(__LOG_ARGS__, 1, "Rank %d: key %s, val: %" PRIu64 "\n", rank_, key,
            val);
    // must use iterators because Metric class has const variables,
    // and therefore can not be assigned to and all
    auto it = map.find(key);
    if (it == map.end()) {
      map.insert({key, Metric(key, rank_)});
      it = map.find(key);
    }

    it->second.LogInstance(val);
  }

  StringVec GetCommonMetrics() {
    StringVec local_metrics;
    for (auto& kv : times_us_) {
      local_metrics.push_back(kv.first);
    }

    auto intersection_computer = CommonComputer(local_metrics, rank_, nranks_);
    return intersection_computer.Compute();
  }

  std::string CollectMetricSummary(StringVec const& metric_vec, int top_k) {
    std::string all_metric_summary;

    // First, need to get metrics that are logged on all ranks
    // as collectives will block on ranks that are missing a given metric
    auto all_metric_stats = Metric::CollectMetrics(metric_vec, times_us_);

    all_metric_summary += MetricPrintUtils::SortAndSerialize(all_metric_stats, top_k);
    all_metric_summary += "\n\n";

    all_metric_summary += p2p_comm_.CollectAndAnalyze(rank_, nranks_);

    return all_metric_summary;
  }

  void LogMetrics() {
    StringVec common_metrics = GetCommonMetrics();
    auto metric_summary = CollectMetricSummary(common_metrics, amr_opts.print_topk);

    if (rank_ == 0) {
      fprintf(stderr, "%s", metric_summary.c_str());
    }

    if (amr_opts.rankwise_enabled) {
      CollectMetricsDetailed(common_metrics);
    }
  }

  void CollectMetricsDetailed(StringVec const& metrics) {
    pdlfs::WritableFile* f;
    pdlfs::Status s =
        env_->NewWritableFile(amr_opts.rankwise_fpath.c_str(), &f);
    if (!s.ok()) {
      Warn(__LOG_ARGS__, "Failed to open file %s",
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
};
}  // namespace amr
