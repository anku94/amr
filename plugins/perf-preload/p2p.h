#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace amr {
struct MatrixStat {
 private:
  uint64_t sum;
  uint64_t sum_sq;
  uint64_t count; // used to compute mean/std

 public:
  MatrixStat() : sum(0), sum_sq(0), count(0) {}

  void Add(uint64_t val) {
    if (val == 0) {
      return;
    }

    sum += val;
    sum_sq += val * val;
    count++;
  }

  uint64_t GetSum() const { return sum; }

  uint64_t GetCount() const { return count; }

  void GetMeanAndStd(double& mean, double& std) const {
    if (count == 0) {
      mean = 0;
      std = 0;
      return;
    }

    mean = static_cast<double>(sum) / count;
    double mean_sq = static_cast<double>(sum_sq) / count;
    std = sqrt(mean_sq - mean * mean);
  }
};

// Two kinds of MatrixAnalysis:
// 1. For the message sizes
// 2. For the message counts
struct MatrixAnalysis {
  MatrixStat local;          // local message size or count
  MatrixStat global;         // global message size or count
  MatrixStat npeers_local;   // degree of the communication graph
  MatrixStat npeers_global;  // degree of the communication graph
};

class P2PCommCollector {
 public:
  P2PCommCollector() : my_rank_(-1), nranks_(-1), npernode_(-1) {}

  void LogSend(int dest, int msg_sz) {
    send_count_[dest]++;
    send_sz_[dest] += msg_sz;
  }

  void LogRecv(int src, int msg_sz) {
    recv_count_[src]++;
    recv_sz_[src] += msg_sz;
  }

  std::string CollectAndAnalyze(int my_rank, int nranks);

 private:
  std::string CollectWithReduce();

  MatrixAnalysis CollectMatrixWithReduce(
      const std::unordered_map<int, uint64_t>& map, bool is_send);

  std::string CollectWithPuts();

  MatrixAnalysis CollectMatrixWithPuts(
      const std::unordered_map<int, uint64_t>& map, bool is_send);

  std::unordered_map<int, uint64_t> send_count_;
  std::unordered_map<int, uint64_t> send_sz_;
  std::unordered_map<int, uint64_t> recv_count_;
  std::unordered_map<int, uint64_t> recv_sz_;

  int my_rank_;
  int nranks_;
  int npernode_;
};
}  // namespace amr
