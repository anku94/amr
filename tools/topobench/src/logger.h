//
// Created by Ankush J on 4/12/22.

#pragma once

#include "common.h"
#include <chrono>
#include <memory>
#include <vector>

using TimePoint = std::chrono::time_point<std::chrono::steady_clock,
                                          std::chrono::duration<double>>;

class MeshBlock;

class Logger {
public:
  Logger()
      : start_ms_{}, end_ms_{}, total_sent_(0), total_rcvd_(0), total_time_(0),
        num_obs_(0) {}

  void LogBegin() { start_ms_ = Now(); }

  void LogEnd() {
    end_ms_ = Now();

    logv(__LOG_ARGS__, LOG_DBUG, "Total time: %.2f us",
         (end_ms_ - start_ms_) * 1e3);

    num_obs_++;
  }

  void LogData(std::vector<std::shared_ptr<MeshBlock>> &blocks_);

  void Aggregate();

  void LogRun(double send_mb, double send_mbps, double recv_mb,
              double recv_mbps, double time_avg_ms, double time_min_ms,
              double time_max_ms, int num_obs);

private:
  int GetNumRanks() const;

  uint64_t Now() const {
    // https://stackoverflow.com/questions/31255486/c-how-do-i-convert-a-stdchronotime-point-to-long-and-back
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::time_point_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now())
                   .time_since_epoch())
        .count();
  }
  uint64_t start_ms_, end_ms_;
  uint64_t total_sent_;
  uint64_t total_rcvd_;
  double total_time_;
  uint64_t num_obs_;
};
