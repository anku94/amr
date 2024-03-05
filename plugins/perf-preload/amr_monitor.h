#include "common.h"
#include "logging.h"
#include "metric.h"

#include <pdlfs-common/env.h>
#include <unordered_map>

namespace amr {
typedef std::unordered_map<std::string, Metric> MetricMap;

class AMRMonitor {
 public:
  AMRMonitor(pdlfs::Env* env, int rank) : env_(env), rank_(rank) {
    Verbose(__LOG_ARGS__, 0, "AMRMonitor initialized on rank %d", rank);
  }

  ~AMRMonitor() {
    Collect();
    Verbose(__LOG_ARGS__, 0, "AMRMonitor destroyed on rank %d", rank_);
  }

  uint64_t Now() const {
    auto now = env_->NowMicros();
    return now;
  }

  void LogMPICollective(const char* type, uint64_t time_us) {
    LogKey(collective_times_us_, type, time_us);
  }

  void LogBegin(const char* key) { begin_times_us_[key] = Now(); }

  void LogEnd(const char* key) {
    auto end_time = Now();

    if (begin_times_us_.find(key) == begin_times_us_.end()) {
      Warn(__LOG_ARGS__, "key %s not found in begin_times_us_.", key);
      return;
    }

    auto begin_time = begin_times_us_[key];
    auto elapsed_time = end_time - begin_time;

    LogKey(key, elapsed_time);

    begin_times_us_.erase(key);
  }

  void LogKey(const char* key, uint64_t val) {
    // split key by delimiter into a vector of strings
    auto tokens = split_string(key, '/');
    if (tokens.size() == 2 && tokens[0] == "kreg") {
      LogKey(kokkos_reg_times_us_, tokens[1].c_str(), val);
    } else if (tokens.size() == 2 && tokens[0] == "mpicoll") {
      LogKey(collective_times_us_, tokens[1].c_str(), val);
    } else {
      Warn(__LOG_ARGS__,
           "Error: key %s not recognized. Should suffer silently.", key);
    }
  }

  static void LogKey(MetricMap& map, const char* key, uint64_t val) {
    if (map.find(key) == map.end()) {
      map[key] = Metric();
    }

    map[key].LogInstance(val);
  }

  void Collect() {
    PMPI_Barrier(MPI_COMM_WORLD);

    if (rank_ == 0) {
      Metric::LogHeader();
    }

    for (auto& kv : collective_times_us_) {
      kv.second.Collect(kv.first.c_str(), rank_);
    }

    for (auto& kv : kokkos_reg_times_us_) {
      kv.second.Collect(kv.first.c_str(), rank_);
    }

    PMPI_Barrier(MPI_COMM_WORLD);
  }

 private:
  MetricMap collective_times_us_;
  MetricMap kokkos_reg_times_us_;

  std::unordered_map<std::string, uint64_t> begin_times_us_;

  pdlfs::Env* const env_;
  const int rank_;
};
}  // namespace amr
