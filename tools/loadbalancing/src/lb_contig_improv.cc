//
// Created by Ankush J on 5/22/23.
//

#include "common.h"
#include "lb_policies.h"

#include <queue>
#include <vector>

namespace {
void GetRollingSum(std::vector<double> const& v, std::vector<double>& sum,
                   int k) {
  double rolling_sum = 0;

  for (int i = 0; i < v.size(); i++) {
    rolling_sum += v[i];

    if (i >= k - 1) {
      sum.push_back(rolling_sum);
      rolling_sum -= v[i - k + 1];
    }
  }

  double rolling_max = *std::max_element(sum.begin(), sum.end());
  double rolling_min = *std::min_element(sum.begin(), sum.end());

  logf(LOG_DBG2, "K: %d, Rolling Max: %.2lf, Rolling Min: %.2lf", k,
       rolling_max, rolling_min);
}

bool IsRangeAvailable(std::vector<int> const& ranklist, int start, int end) {
  for (int i = start; i <= end; i++) {
    if (ranklist[i] != -1) {
      return false;
    }
  }
  return true;
}

bool MarkRange(std::vector<int>& ranklist, int start, int end, int flag) {
  logf(LOG_DBG2, "MarkRange marking (%d, %d) with %d", start, end, flag);

  for (int i = start; i <= end; i++) {
    ranklist[i] = flag;
  }
  return true;
}

int AssignBlocksContigImproved(std::vector<double> const& costlist,
                               std::vector<int>& ranklist, int nranks) {
  double nperrank = costlist.size() / nranks;
  double cost_total = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_target = cost_total / nranks;
  logf(LOG_DBG2, "Target Cost: %.2lf", cost_target);

  std::vector<double> sum_a, sum_b;
  int n_a = std::floor(costlist.size() * 1.0 / nranks);
  int n_b = std::ceil(costlist.size() * 1.0 / nranks);
  int nalloc_b = costlist.size() % nranks;
  int nalloc_a = nranks - nalloc_b;

  GetRollingSum(costlist, sum_a, n_a);
  GetRollingSum(costlist, sum_b, n_b);

  // do nalloc_b allocs of size n_b
  std::fill(ranklist.begin(), ranklist.end(), -1);
  std::priority_queue<std::pair<double, int>,
                      std::vector<std::pair<double, int>>,
                      std::greater<std::pair<double, int>>>
      pq;

  for (int i = 0; i < sum_b.size(); i++) {
    pq.push(std::make_pair(sum_b[i], i));
  }

  int b_alloc_count = 0;
  while (b_alloc_count < nalloc_b) {
    auto top = pq.top();
    pq.pop();
    int idx = top.second;
    // This cost represents a range of n_b items starting from idx
    // The range would be [idx, idx + n_b - 1]
    if (!IsRangeAvailable(ranklist, idx, idx + n_b - 1)) {
      continue;
    }

    logf(LOG_DBG2, "Allocating n_b range %d to %d (cost: %.1lf)", idx,
         idx + n_b - 1, top.first);

    MarkRange(ranklist, idx, idx + n_b - 1, -2);
    b_alloc_count++;
  }

  int cur_rank = 0;
  int prev_flag = 0;
  int cur_range_beg = INT_MIN;
  int nmax_prev = 0;

  for (int cur_idx = 0; cur_idx < ranklist.size(); cur_idx++) {
    if ((ranklist[cur_idx] != prev_flag) or
        (cur_idx - cur_range_beg >= nmax_prev)) {
      if (cur_idx > 0) {
        MarkRange(ranklist, cur_range_beg, cur_idx - 1, cur_rank);
        cur_rank++;
      }

      prev_flag = ranklist[cur_idx];
      cur_range_beg = cur_idx;
      nmax_prev = (prev_flag == -1) ? n_a : n_b;
    }
  }

  MarkRange(ranklist, cur_range_beg, ranklist.size() - 1, cur_rank++);
  assert(cur_rank == nranks);

  return 0;
}
}  // namespace

namespace amr {
int LoadBalancePolicies::AssignBlocksContigImproved(
    std::vector<double> const& costlist, std::vector<int>& ranklist,
    int nranks) {
  return ::AssignBlocksContigImproved(costlist, ranklist, nranks);
}
}  // namespace amr