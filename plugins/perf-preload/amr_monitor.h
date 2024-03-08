#pragma once

#include "common.h"
#include "logging.h"
#include "metric.h"
#include "p2p.h"

#include <pdlfs-common/env.h>
#include <stack>
#include <unordered_map>

namespace amr {
typedef std::unordered_map<std::string, Metric> MetricMap;
typedef std::pair<std::string, uint64_t> StackOpenPair;
typedef std::stack<StackOpenPair> ProfStack;
typedef std::unordered_map<std::string, ProfStack> StackMap;

class AMRMonitor {
 public:
  AMRMonitor(pdlfs::Env* env, int rank, int nranks)
      : env_(env), rank_(rank), nranks_(nranks) {
    Verbose(__LOG_ARGS__, 1, "AMRMonitor initialized on rank %d", rank);
  }

  ~AMRMonitor() {
    LogAllMetrics();
    Verbose(__LOG_ARGS__, 1, "AMRMonitor destroyed on rank %d", rank_);
  }

  uint64_t Now() const {
    auto now = env_->NowMicros();
    return now;
  }
  void LogMPICollective(const char* type, uint64_t time_us) {
    LogKey(collective_times_us_, type, time_us);
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
    LogKey(kokkos_reg_times_us_, key.c_str(), elapsed_time);

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
    std::string key = "MPI_Send_" + std::to_string(dest_rank);
    auto size = count * GetMPITypeSizeCached(datatype);

    LogKey(mpi_comm_msgsz_, key.c_str(), size);
    p2p_comm_.LogSend(dest_rank, size);
  }

  void LogMPIRecv(int src_rank, MPI_Datatype datatype, int count) {
    std::string key = "MPI_Recv_" + std::to_string(src_rank);
    auto size = count * GetMPITypeSizeCached(datatype);

    LogKey(mpi_comm_msgsz_, key.c_str(), size);
    p2p_comm_.LogRecv(src_rank, size);
  }

  static void LogKey(MetricMap& map, const char* key, uint64_t val) {
    if (map.find(key) == map.end()) {
      map[key] = Metric();
    }

    map[key].LogInstance(val);
  }

  std::string CollectAllMetrics() {
    std::string metrics;

    for (auto& kv : collective_times_us_) {
      metrics += kv.second.Collect(kv.first.c_str(), rank_);
    }

    for (auto& kv : kokkos_reg_times_us_) {
      metrics += kv.second.Collect(kv.first.c_str(), rank_);
    }

    metrics += p2p_comm_.CollectAndAnalyze(rank_, nranks_);

    return metrics;
  }

  void LogAllMetrics() {
    auto metrics = CollectAllMetrics();

    if (rank_ == 0) {
      auto header = MetricPrintUtils::GetHeader();
      fprintf(stderr, "%s", metrics.c_str());
      fprintf(stderr, "%s", header.c_str());
    }
  }

 private:
  MetricMap collective_times_us_;
  MetricMap mpi_comm_msgsz_;
  MetricMap kokkos_reg_times_us_;
  StackMap stack_map_;

  P2PCommCollector p2p_comm_;

  std::unordered_map<MPI_Datatype, uint32_t> mpi_datatype_sizes_;

  std::unordered_map<std::string, uint64_t> begin_times_us_;

  pdlfs::Env* const env_;
  const int rank_;
  const int nranks_;
};
}  // namespace amr
