//
// Created by Ankush J on 4/12/22.

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "amr/block.h"
#include "tools-common/logging.h"
#include "metric_utils.h"

namespace topo {
// using ExtraMetric = std::pair<std::string, std::string>;
// using ExtraMetricVec = std::vector<ExtraMetric>;

//
// Logger: log stats for a communication round
//
class Logger {
 public:
  Logger() = default;

  // LogBegin: log the start time of the communication round
  void LogBegin() { start_us_ = NowMicros(); }

  // LogEnd: log the total time taken for the communication round
  void LogEnd() {
    end_us_ = NowMicros();
    auto dur_us = end_us_ - start_us_;
    MLOGIFR0(MLOG_INFO, "- Last round time: %" PRIu64 " us", dur_us);
    num_obs_++;
  }

  // LogData: drain bytes sent/rcvd from blocks
  void DrainBlockData(std::vector<std::shared_ptr<topo::MeshBlock>> &blocks);

  // Aggregate: gather all stats using collectives, and print/log them
  // - Creates one row in the log csv (a "run")
  void AggregateAndWrite(ExtraMetricVec &extra_metrics, const char *log_fpath);

 private:
  // GetNumRanks: uses MPI_Comm_size to get the number of ranks
  // Not sure if needed
  int GetNumRanks() const;

  // NowMicros: get the current time in microseconds
  uint64_t NowMicros() const {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
  }

  uint64_t start_us_{0};
  uint64_t end_us_{0};
  uint64_t totbytes_sent_{0};
  uint64_t totbytes_rcvd_{0};
  uint64_t totcnt_sent_{0};
  uint64_t totcnt_rcvd_{0};
  double totdur_ms_{0};
  uint64_t num_obs_{0};
};
}  // namespace topo
