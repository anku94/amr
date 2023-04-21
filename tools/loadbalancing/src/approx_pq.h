//
// Created by Ankush J on 4/13/23.
//

#pragma once

namespace amr {
struct ApproxPQItem {
  int rank;
  double load;
};

class ApproxPQBucket {
 private:
  double load_class_;
  std::deque<ApproxPQItem> q_;
};
/*
 * Approximate Priority Queue for approx LPT allocation
 */
class ApproxPQ {
 public:
  explicit ApproxPQ(int delta) : delta_(delta) {}

  void Seed(int nranks) {
    for (int rank = nranks - 1; rank >= 0; rank--) {
      q_.push_back({rank, 0});
    }
  }

 private:
  std::deque<ApproxPQItem> q_;
  int delta_;
};
};  // namespace amr