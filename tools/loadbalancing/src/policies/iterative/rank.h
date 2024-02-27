//
// Created by Ankush J on 7/4/23.
//

#pragma once

#include <algorithm>

namespace amr {
class Rank {
 public:
  Rank(int rank) : sorted_(false), rank_(rank), cost_(0) {}

  void AddBlock(int bidx, double cost) {
    block_vec_.push_back(std::make_pair(cost, bidx));
    cost_ += cost;
    sorted_ = false;
  }

  void Sort() {
    std::sort(block_vec_.begin(), block_vec_.end());
    sorted_ = true;
  }

  double GetCost() const { return cost_; }

  void GetLargestBlock(int& bidx, double& cost) {
    if (!sorted_) {
      Sort();
    }

    if (block_vec_.empty()) {
      ABORT("No blocks in rank");
    }

    bidx = block_vec_.back().second;
    cost = block_vec_.back().first;
  }

  void GetSmallestBlock(int& bidx, double& cost) {
    if (!sorted_) {
      Sort();
    }

    if (block_vec_.empty()) {
      ABORT("No blocks in rank");
    }

    bidx = block_vec_.front().second;
    cost = block_vec_.front().first;
  }

  double RemoveBlock(int bidx) {
    int found = -1;

    for (int i = 0; i < block_vec_.size(); i++) {
      if (block_vec_[i].second == bidx) {
        found = i;
        break;
      }
    }

    if (found != -1) {
      double block_cost = block_vec_[found].first;
      cost_ -= block_cost;
      block_vec_.erase(block_vec_.begin() + found);
      return block_cost;
    } else {
      ABORT("Block not found");
    }

    return 0;
  }

  void GetBlockLoc(std::vector<std::pair<int, int>>& br_vec) {
    for (auto& block : block_vec_) {
      br_vec.emplace_back(block.second, rank_);
    }
  }

  bool HasBlocks() const {
    return !block_vec_.empty();
  }

  const int rank_;

 private:
  bool sorted_;
  double cost_;
  std::vector<std::pair<double, int>> block_vec_;
};
}  // namespace amr
