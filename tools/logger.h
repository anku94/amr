//
// Created by Ankush J on 4/12/22.
//

#pragma once

#include <chrono>
#include <vector>

using TimePoint = std::chrono::time_point<std::chrono::steady_clock,
                                          std::chrono::duration<double>>;

class MeshBlock;

class Logger {
 public:
  Logger() : start_{}, end_{}, total_sent_(0), total_rcvd_(0), total_time_(0) {}

  void LogBegin() { start_ = std::chrono::high_resolution_clock::now(); }

  void LogEnd() { end_ = std::chrono::high_resolution_clock::now(); }

  void LogData(std::vector<std::shared_ptr<MeshBlock>>& blocks_);

  void Aggregate();

 private:
  TimePoint start_, end_;
  uint64_t total_sent_;
  uint64_t total_rcvd_;
  double total_time_;
};