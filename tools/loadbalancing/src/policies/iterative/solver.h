//
// Created by Ankush J on 7/4/23.
//

#pragma once

#include "rank.h"

#include <algorithm>
#include <cassert>
#include <queue>

namespace amr {
class Solver {
 public:
  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks, int niters) {
    nranks_ = nranks;
    double avg_cost, max_cost;

    InitializeRanks(nranks, costlist, ranklist);
    GetRankStats(ranks_, avg_cost, max_cost);
    LogRankStats("INITIAL", avg_cost, max_cost);

    for (int iter = 0; iter < niters; iter++) {
      Iterate();
      GetRankStats(ranks_, avg_cost, max_cost);
      LogRankStats(iter, avg_cost, max_cost);
    }

    LogRankStats("FINAL", avg_cost, max_cost);

    UpdateRanklist(ranklist);
    return 0;
  }

  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks,
                   const double target_cost, int& niters,
                   const int max_iters = kMaxIters) {
    nranks_ = nranks;
    double avg_cost, max_cost;

    InitializeRanks(nranks, costlist, ranklist);
    GetRankStats(ranks_, avg_cost, max_cost);
    LogRankStats("INITIAL", avg_cost, max_cost);

    niters = 0;
    while (max_cost > target_cost) {
      Iterate();
      GetRankStats(ranks_, avg_cost, max_cost);
      LogRankStats(niters, avg_cost, max_cost);
      niters++;

      if (niters == max_iters) {
        logf(LOG_WARN, "Solver hit kMaxIters (%d). Ending prem...", max_iters);
        break;
      }
    }

    LogRankStats("FINAL", avg_cost, max_cost);

    UpdateRanklist(ranklist);
    return 0;
  }

  static void AnalyzePlacement(std::vector<double> const& costlist,
                               std::vector<int>& ranklist, int nranks,
                               double& avg_cost, double& max_cost) {
    std::vector<Rank> ranks;
    InitializeRanks(ranks, nranks, costlist, ranklist);
    GetRankStats(ranks, avg_cost, max_cost);
  }

 private:
  void InitializeRanks(int nranks, std::vector<double> const& costlist,
                       std::vector<int> const& ranklist) {
    ranks_.clear();
    rank_cost_vec_.clear();

    InitializeRanks(ranks_, nranks_, costlist, ranklist);

    for (auto& rank : ranks_) {
      rank_cost_vec_.emplace_back(rank.GetCost(), rank.rank_);
    }
  }

  static void InitializeRanks(std::vector<Rank>& ranks, int nranks,
                              std::vector<double> const& costlist,
                              std::vector<int> const& ranklist) {
    ranks.clear();

    for (int rank_id = 0; rank_id < nranks; rank_id++) {
      ranks.emplace_back(rank_id);
    }

    for (int block_idx = 0; block_idx < costlist.size(); block_idx++) {
      int rank = ranklist[block_idx];
      if (rank >= ranks.size()) {
        ABORT("Rank > nranks_");
      }

      ranks[rank].AddBlock(block_idx, costlist[block_idx]);
    }
  }

  void Iterate() {
    std::sort(rank_cost_vec_.begin(), rank_cost_vec_.end());

    int lb_bidx, sb_bidx, lb_rank, sb_rank;
    double lb_cost, sb_cost;

    lb_rank = rank_cost_vec_.back().second;
    ranks_[lb_rank].GetLargestBlock(lb_bidx, lb_cost);

    sb_rank = rank_cost_vec_.front().second;
    ranks_[sb_rank].GetSmallestBlock(sb_bidx, sb_cost);

    logf(LOG_DBG2, "Before. r%d: %.0lf, r%d: %.0lf", sb_rank,
         ranks_[sb_rank].GetCost(), lb_rank, ranks_[lb_rank].GetCost());
    logf(LOG_DBG2,
         "Swapping blocks (c%.0lf, r%d) and (c%.0lf, r%d). (Reduction: %.0lf)",
         sb_cost, sb_rank, lb_cost, lb_rank, lb_cost - sb_cost);

    TransferBlock(lb_bidx, lb_rank, sb_rank, nranks_ - 1, 0);
    TransferBlock(sb_bidx, sb_rank, lb_rank, 0, nranks_ - 1);

    logf(LOG_DBG2, "After. r%d: %.0lf, r%d: %.0lf", sb_rank,
         ranks_[sb_rank].GetCost(), lb_rank, ranks_[lb_rank].GetCost());
  }

  void TransferBlock(int bidx, int src_rank, int dest_rank, int src_ridx,
                     int dest_ridx) {
    double cost = ranks_[src_rank].RemoveBlock(bidx);
    ranks_[dest_rank].AddBlock(bidx, cost);

    assert(rank_cost_vec_[src_ridx].second == src_rank);
    assert(rank_cost_vec_[dest_ridx].second == dest_rank);

    rank_cost_vec_[src_ridx].first -= cost;
    rank_cost_vec_[dest_ridx].first += cost;
  }

  static void LogRankStats(int iter, double& avg_cost, double& max_cost) {
    char buf[128];
    snprintf(buf, 128, "ITER %d", iter);
    LogRankStats(buf, avg_cost, max_cost);
  }

  static void LogRankStats(const char* prefix, double avg_cost,
                           double max_cost) {
    logf(LOG_DBUG, "[%s] Rank Stats: Avg Cost: %.0f, Max: %.0f\n", prefix,
         avg_cost, max_cost);
  }

  static void GetRankStats(std::vector<Rank>& ranks, double& avg_cost,
                           double& max_cost) {
    avg_cost = max_cost = 0;

    for (auto& r : ranks) {
      avg_cost += r.GetCost();
      max_cost = std::max(max_cost, r.GetCost());
    }

    avg_cost /= ranks.size();
  }

  void UpdateRanklist(std::vector<int>& ranklist) {
    std::vector<std::pair<int, int>> br_vec;  // block-rank
    for (auto& r : ranks_) {
      r.GetBlockLoc(br_vec);
    }

    if (br_vec.size() != ranklist.size()) {
      ABORT("br_vec.size() != ranklist.size()");
    }

    std::sort(br_vec.begin(), br_vec.end());
    for (int i = 0; i < br_vec.size(); i++) {
      ranklist[i] = br_vec[i].second;
    }
  }

  int nranks_;
  std::vector<Rank> ranks_;
  std::vector<std::pair<double, int>> rank_cost_vec_;

 public:
  static constexpr int kMaxIters = 250;
};
}  // namespace amr
