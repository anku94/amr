//
// Created by Ankush J on 5/4/23.
//

#pragma once

#include <map>

namespace amr {
class CostCache {
 public:
  explicit CostCache(int ttl = 5)
      : ttl_(ttl), req_cnt_(0), req_miss_(0), req_exp_(0) {}

  ~CostCache() { LogStats(); }

  bool Get(int ts, int nblocks, std::vector<double>& cost) {
    req_cnt_++;

    auto it = cache_.find(nblocks);
    if (it == cache_.end()) {
      req_miss_++;
      return false;
    }

    if (ts - it->second.first > ttl_) {
      req_exp_++;
      return false;
    }

    cost = it->second.second;
    return true;
  }

  void Put(int ts, std::vector<double> const& cost) {
    int nblocks = cost.size();
    cache_[nblocks] = std::make_pair(ts, cost);
  }

  void LogStats() const {
    if (req_cnt_ == 0) return;

    logf(LOG_INFO,
         "[CostCache] Requests: %d, Hits: %d (Misses: %d, Expired: %d)",
         req_cnt_, req_cnt_ - req_miss_ - req_exp_, req_miss_, req_exp_);
  }

 private:
  const int ttl_;
  std::map<int, std::pair<int, std::vector<double>>> cache_;

  int req_cnt_;   // Total requests
  int req_miss_;  // Not found in cache
  int req_exp_;   // Found in cache but expired
};
}  // namespace amr