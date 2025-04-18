//
// Created by Ankush J on 4/12/22.

#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "amr/block.h"
#include "logging.h"

namespace topo {
using ExtraMetric = std::pair<std::string, std::string>;
using ExtraMetricVec = std::vector<ExtraMetric>;

//
// Logger: log stats for a communication round
//
class Logger {
 public:
  Logger()
      : start_us_{},
        end_us_{},
        totbytes_sent_(0),
        totbytes_rcvd_(0),
        totdur_ms_(0),
        num_obs_(0) {}

  // LogBegin: log the start time of the communication round
  void LogBegin() { start_us_ = NowMicros(); }

  // LogEnd: log the total time taken for the communication round
  void LogEnd() {
    end_us_ = NowMicros();
    MLOGIFR0(MLOG_INFO, "Total time: %.2f us", (end_us_ - start_us_));
    num_obs_++;
  }

  // LogData: drain bytes sent/rcvd from blocks
  void DrainBlockData(std::vector<std::shared_ptr<topo::MeshBlock>> &blocks);

  // Aggregate: gather all stats using collectives, and print/log them
  // - Creates one row in the log csv (a "run")
  void AggregateAndWrite(ExtraMetricVec &extra_metrics, const char *log_fpath);

 private:
  // LogRun: add a run row to the log csv, called within Aggregate
  // void LogRun();

  // static std::vector<std::string> GetHeader();

  int GetNumRanks() const;

  uint64_t NowMicros() const {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
  }

  uint64_t start_us_, end_us_;
  uint64_t totbytes_sent_;
  uint64_t totbytes_rcvd_;
  double totdur_ms_;
  uint64_t num_obs_;
};
}  // namespace topo
