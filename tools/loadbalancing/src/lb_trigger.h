//
// Created by Ankush J on 4/17/23.
//

#pragma once

namespace amr {
class LoadBalanceTrigger {
 public:
  explicit LoadBalanceTrigger(int nranks)
      : nranks_(nranks), ts_(0), ts_lb_(0) {}

  LoadBalanceTrigger(LoadBalanceTrigger&& rhs) noexcept
      : nranks_(rhs.nranks_),
        ts_(rhs.ts_),
        ts_lb_(rhs.ts_lb_),
        ranklist_prev_(std::move(rhs.ranklist_prev_)) {}

  LoadBalanceTrigger(const LoadBalanceTrigger& rhs) = delete;

  LoadBalanceTrigger& operator=(LoadBalanceTrigger&& rhs) = delete;

  bool Trigger(std::vector<double> const& costlist) {
    bool trigger = false;

    if (costlist.size() != ranklist_prev_.size()) {
      trigger = true;
    } else {
      std::vector<double> ranks(nranks_, 0);
      Allocate(ranks, costlist, ranklist_prev_);
      double rank_avg = std::accumulate(ranks.begin(), ranks.end(), 0.0) /
                        static_cast<double>(nranks_);
      double rank_max = *std::max_element(ranks.begin(), ranks.end());
      if (rank_max > rank_avg * 1.2) trigger = true;
    }

    if (trigger) ts_lb_++;
    ts_++;
    return trigger;
  }

  void Update(std::vector<int>& costlist, std::vector<int>& ranklist) {
    ranklist_prev_ = ranklist;
  }

  std::vector<int> GetLastAssignment() const { return ranklist_prev_; }

 private:
  static void Allocate(std::vector<double>& ranks,
                       std::vector<double> const& costlist,
                       std::vector<int>& ranklist) {
    for (int i = 0; i < costlist.size(); ++i) {
      ranks[ranklist[i]] += costlist[i];
    }
  }

  int nranks_;
  int ts_;
  int ts_lb_;

  std::vector<int> ranklist_prev_;
};
}  // namespace amr