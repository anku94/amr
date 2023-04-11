//
// Created by Ankush J on 4/10/23.
//

#include "lb_policies.h"

#include <numeric>
#include <queue>
#include <vector>

/*
 * LongestProcessingTime or ShortestProcessingTime
 */

namespace {
class Rank {
 public:
  int id;
  double load;

  Rank(int id, double load) : id(id), load(load) {}
};

struct RankComparator {
  bool operator()(const Rank& a, const Rank& b) { return a.load > b.load; }
};

template <typename Comparator>
void AssignBlocks(std::vector<double> const& costlist,
                  std::vector<int>& ranklist, int nranks, Comparator comp) {
  // Initialize the ranklist with -1
  std::fill(ranklist.begin(), ranklist.end(), -1);

  // Create a priority queue of ranks with the load as the priority
  std::priority_queue<Rank, std::vector<Rank>, RankComparator> rankQueue;
  for (int i = 0; i < nranks; i++) {
    rankQueue.push(Rank(i, 0.0));
  }

  // Sort the indices of the costlist in ascending order of their corresponding
  // costs
  std::vector<int> indices(costlist.size());
  std::iota(indices.begin(), indices.end(), 0);
  //  std::sort(indices.begin(), indices.end(),
  //            [&](int a, int b) { return costlist[a] < costlist[b]; });
  std::sort(indices.begin(), indices.end(), comp);

  // Assign the blocks to the ranks using SPT algorithm
  for (int idx : indices) {
    Rank minLoadRank = rankQueue.top();
    rankQueue.pop();

    ranklist[idx] = minLoadRank.id;
    minLoadRank.load += costlist[idx];
    rankQueue.push(minLoadRank);
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
