//
// Created by Ankush J on 4/10/23.
//

#include <algorithm>
#include <numeric>
#include <queue>
#include <vector>

#include "common.h"
#include "lb_policies.h"
#include "lb_util.h"

/*
 * LongestProcessingTime or ShortestProcessingTime
 */

namespace {
template <typename Comparator>
void AssignBlocks(std::vector<double> const& costlist,
                  std::vector<int>& ranklist, int nranks, Comparator comp) {
  // Initialize the ranklist with -1
  ranklist.resize(costlist.size());
  std::fill(ranklist.begin(), ranklist.end(), -1);

  // Create a priority queue of ranks with the load as the priority
  std::priority_queue<Rank, std::vector<Rank>, LeastLoadedRankPQComparator>
      rank_queue;

  const float delta = 0.0001;
  for (int i = 0; i < nranks; i++) {
    // bootstrap with small load to prefer order
    rank_queue.emplace(i, delta * i);
    // rank_queue.push(Rank(i, 0.0));
  }

  // Sort the indices of the costlist_prev in ascending order of their
  // corresponding costs
  std::vector<int> indices(costlist.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), comp);

  // Assign the blocks to the ranks using SPT algorithm
  for (int idx : indices) {
    Rank minLoadRank = rank_queue.top();
    rank_queue.pop();

    logv(__LOG_ARGS__, LOG_DBG3,
         "[LPT] Block (%d, %.1f) assigned to rank (%d, %.1f)", idx,
         costlist[idx], minLoadRank.id, minLoadRank.load);

    ranklist[idx] = minLoadRank.id;
    minLoadRank.load += costlist[idx];
    rank_queue.push(minLoadRank);
  }
}
}  // namespace

namespace amr {
int LoadBalancePolicies::AssignBlocksSPT(std::vector<double> const& costlist,
                                         std::vector<int>& ranklist,
                                         int nranks) {
  ::AssignBlocks(costlist, ranklist, nranks,
                 [&](int a, int b) { return costlist[a] < costlist[b]; });
  return 0;
}

int LoadBalancePolicies::AssignBlocksLPT(std::vector<double> const& costlist,
                                         std::vector<int>& ranklist,
                                         int nranks) {
  ::AssignBlocks(costlist, ranklist, nranks,
                 [&](int a, int b) { return costlist[a] > costlist[b]; });
  return 0;
}
}  // namespace amr
