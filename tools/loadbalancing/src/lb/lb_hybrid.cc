#include <algorithm>
#include <cassert>
#include <queue>

#include "common.h"
#include "iterative/solver.h"
#include "lb_policies.h"
#include "lb_util.h"
#include "policy_utils.h"
#include "policy_wopts.h"

typedef std::pair<int, double> BlockCostPair;

class HybridAssignment {
 public:
  HybridAssignment(int num_lpt)
      : num_lpt_(num_lpt), unassigned_first_(-1), unassigned_last_(-1) {
    logv(__LOG_ARGS__, LOG_DBUG, "[HybridPolicy] num_lpt: %d", num_lpt_);
  }

  int AssignBlocksLPT(const int nblocks, const int nranks,
                      const float target_load);

  int AssignBlocksRest(int nranks_rest);

  int AssignBlocks(const std::vector<double>& costlist,
                   std::vector<int>& ranklist, int nranks);

  void AssertAllAssigned() const {
    for (int i = 0; i < ranklist_.size(); i++) {
      if (ranklist_[i] == -1) {
        logv(__LOG_ARGS__, LOG_ERRO, "Block %d not assigned", i);
        ABORT("Some blocks not assigned");
      }
    }
  }

  void LogBlockCostVector(std::vector<BlockCostPair>& block_costs) {
    for (auto& bc : block_costs) {
      logv(__LOG_ARGS__, LOG_INFO, "Block: %d, Cost: %f", bc.first, bc.second);
    }
  }

  double GetLPTMax(std::vector<double> const& costlist, int nranks) {
    std::vector<int> ranklist(nranks, -1);
    auto& lpt_policy = amr::PolicyUtils::GetPolicy("lpt");
    int rv = amr::LoadBalancePolicies::AssignBlocks(lpt_policy, costlist,
                                                    ranklist, nranks);
    if (rv) {
      ABORT("AssignBlocks LPT failed");
    }

    std::vector<double> rank_loads(nranks, 0);
    for (int i = 0; i < costlist.size(); i++) {
      rank_loads[ranklist[i]] += costlist[i];
    }

    auto max_load = *std::max_element(rank_loads.begin(), rank_loads.end());
    return max_load;
  }

 private:
  const int num_lpt_;

  std::vector<BlockCostPair> block_costs_;
  std::vector<int> ranklist_;
  int unassigned_first_;
  int unassigned_last_;
};

namespace amr {
int LoadBalancePolicies::AssignBlocksHybrid(const std::vector<double>& costlist,
                                            std::vector<int>& ranklist,
                                            int nranks,
                                            PolicyOptsHybrid const& opts) {
  int nblocks = costlist.size();
  double lpt_threshold = opts.frac_lpt;
  int num_lpt = std::min(lpt_threshold * nblocks, nranks * 0.9);

  logv(__LOG_ARGS__, LOG_INFO, "LPT Threshold: %.2f, Num LPT: %d",
       lpt_threshold, num_lpt);

  auto ha = HybridAssignment(num_lpt);
  int rv = ha.AssignBlocks(costlist, ranklist, nranks);
  return rv;
}
}  // namespace amr

//
// First, assign k blocks using LPT. Then, assign rest using SPT
// until no rank can take more blocks without breaching target_load
// Assumes: block_costs_ is sorted in descending order
// Sets: ranklist_ and unassigned_first_, unassigned_last_
//
int HybridAssignment::AssignBlocksLPT(const int nblocks, const int nranks,
                                      const float target_load) {
  std::priority_queue<Rank, std::vector<Rank>, LeastLoadedRankPQComparator>
      spt_pq;

  // Assign first num_lpt_ blocks using LPT
  for (int idx = 0; idx < num_lpt_; idx++) {
    auto& block = block_costs_[idx];
    spt_pq.push(Rank(idx, block.second));
    ranklist_[block.first] = idx;

    logv(__LOG_ARGS__, LOG_DBG2,
         "[LPT] Block (%d, %.1f) assigned to rank (%d, %.1f)", block.first,
         block.second, idx, block.second);
  }

  unassigned_first_ = num_lpt_;
  unassigned_last_ = block_costs_.size();

  // Assign rest of the blocks using SPT
  for (int idx = nblocks - 1; idx >= num_lpt_; idx--) {
    Rank min_rank = spt_pq.top();
    spt_pq.pop();

    auto& block = block_costs_[idx];
    if (min_rank.load + block.second > target_load) {
      break;
    }

    ranklist_[block.first] = min_rank.id;
    min_rank.load += block.second;
    spt_pq.push(min_rank);
    unassigned_last_ = idx;

    logv(__LOG_ARGS__, LOG_DBG2,
         "[SPT] Block (%d, %.1f) assigned to rank (%d, %.1f)", block.first,
         block.second, min_rank.id, min_rank.load);
  }

  return 0;
}

int HybridAssignment::AssignBlocksRest(int nranks_rest) {
  int rv = 0;
  int nblocks_rest = unassigned_last_ - unassigned_first_;

  // Sort the unallocated slice by index, for locality
  std::sort(block_costs_.begin() + unassigned_first_,
            block_costs_.begin() + unassigned_last_,
            [](const BlockCostPair& a, const BlockCostPair& b) {
              return a.first < b.first;
            });

  std::vector<double> costlist_rest;
  for (int i = unassigned_first_; i < unassigned_last_; i++) {
    costlist_rest.push_back(block_costs_[i].second);
  }

  std::vector<int> ranklist_rest(nblocks_rest, -1);

  auto& cdp_policy = amr::PolicyUtils::GetPolicy("cdp");
  rv = amr::LoadBalancePolicies::AssignBlocks(cdp_policy, costlist_rest,
                                              ranklist_rest, nranks_rest);

  // for each bidx, ridx in ranklist_rest
  // map it to bidx_orig, ridx' in ranklist_orig
  for (int idx = unassigned_first_; idx < unassigned_last_; idx++) {
    int bidx_new = idx - unassigned_first_;
    int bidx_old = block_costs_[idx].first;
    int ridx_new = ranklist_rest[bidx_new];
    int ridx_old = ridx_new + unassigned_first_;

    ranklist_[bidx_old] = ridx_old;
    logv(__LOG_ARGS__, LOG_DBG2, "[Rest] Block (%d, %.1f) assigned to rank %d",
         bidx_old, block_costs_[idx].second, ridx_old);
  }
  return 0;
}

int HybridAssignment::AssignBlocks(const std::vector<double>& costlist,
                                   std::vector<int>& ranklist, int nranks) {
  int rv = 0;

  int nblocks = costlist.size();
  block_costs_.clear();
  ranklist_.clear();
  ranklist_.resize(nblocks, -1);

  // Preserve block-index in costlist before sorting
  for (int i = 0; i < costlist.size(); i++) {
    block_costs_.push_back(std::make_pair(i, costlist[i]));
  }

  // Sort costlist in descending order
  std::sort(block_costs_.begin(), block_costs_.end(),
            [](const BlockCostPair& a, const BlockCostPair& b) {
              return a.second > b.second;
            });

  double cost_total = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_per_rank = cost_total / nranks;
  double blocks_per_rank = costlist.size() * 1.0 / nranks;

  logv(__LOG_ARGS__, LOG_DBUG,
       "[HybridPolicy] Basic stats:\n"
       "\tTotal Cost: %.1f\n"
       "\tCost Per Rank: %.1f\n"
       "\tBlocks Per Rank: %.1f\n",
       cost_total, cost_per_rank, blocks_per_rank);

  double lpt_max = GetLPTMax(costlist, nranks);
  logv(__LOG_ARGS__, LOG_INFO, "[HybridPolicy] LPT Max: %.0lf\n", lpt_max);

  unassigned_first_ = 0;
  unassigned_last_ = block_costs_.size();

  rv = AssignBlocksLPT(nblocks, nranks, lpt_max);
  if (rv) return rv;

  assert(unassigned_first_ == num_lpt_);

  rv = AssignBlocksRest(nranks - num_lpt_);
  if (rv) return rv;

  AssertAllAssigned();

  amr::Solver solver;
  solver.AssignBlocks(costlist, ranklist_, nranks, nranks);
  ranklist_.swap(ranklist);

  return 0;
}
