//
// Created by Ankush J on 4/12/22.
//

#pragma once

#include <chrono>
#include <memory>
#include <vector>

using TimePoint = std::chrono::time_point<std::chrono::steady_clock,
                                          std::chrono::duration<double>>;

class MeshBlock;

class Logger {
 public:
  Logger()
      : start_ms_{},
        end_ms_{},
        total_sent_(0),
        total_rcvd_(0),
        total_time_(0) {}

  void LogBegin() { start_ms_ = Now(); }

  void LogEnd() { end_ms_ = Now(); }

  void LogData(std::vector<std::shared_ptr<MeshBlock>>& blocks_);

  void Aggregate();

  void LogToFile(double send_bw, double recv_bw);

 private:
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
};
