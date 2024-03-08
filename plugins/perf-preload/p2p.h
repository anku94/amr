#pragma once

#include <unordered_map>

namespace amr {
struct MatrixAnalysis {
  uint64_t sum_local;
  uint64_t sum_global;
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
