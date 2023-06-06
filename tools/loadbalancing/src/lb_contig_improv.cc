//
// Created by Ankush J on 5/22/23.
//

#include "common.h"
#include "lb_policies.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
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
  logf(LOG_DBG2, "MarkRange marking [%d, %d] with %d", start, end, flag);

  for (int i = start; i <= end; i++) {
    ranklist[i] = flag;
  }
  return true;
}

int AssignBlocksDP(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks) {
  double cost_total = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_target = cost_total / nranks;
  logf(LOG_DBG2, "Target Cost: %.2lf", cost_target);

  std::vector<double> cum_costlist(costlist);
  int nblocks = costlist.size();
  int n_a = std::floor(nblocks * 1.0 / nranks);
  int n_b = std::ceil(nblocks * 1.0 / nranks);
  int nalloc_b = nblocks % nranks;
  int nalloc_a = nranks - nalloc_b;

  for (int i = 1; i < nblocks; i++) {
    cum_costlist[i] += cum_costlist[i - 1];
  }

  const double kBigDouble = cost_total * 1e3;
  std::vector<double> dp(nblocks + 1, kBigDouble);

  dp[0] = 0;
  for (int i = 1; i <= nblocks; i++) {
    // option 1. n_a blocks ending at i-1
    double opt1_cost = kBigDouble;
    int opt1_start = (i - n_a);  // (i - 1) - n_a + 1
    if (opt1_start >= 0) {
      // cost of the range [opt1_start, i-1]
      // cost of range [a, b] is cum_costlist[b] - cum_costlist[a-1]
      double opt1_cost_a = cum_costlist[i - 1] - cum_costlist[opt1_start - 1];
      double opt1_cost_b = dp[opt1_start];
      opt1_cost = std::max(opt1_cost_a, opt1_cost_b);
    }

    // option 2. n_b blocks ending at i-1
    double opt2_cost = kBigDouble;
    int opt2_start = (i - n_b);
    if (opt2_start >= 0) {
      double opt2_cost_a = cum_costlist[i - 1] - cum_costlist[opt2_start - 1];
      double opt2_cost_b = dp[opt2_start];
      opt2_cost = std::max(opt2_cost_a, opt2_cost_b);
    }

    double dp_cost = std::min(opt1_cost, opt2_cost);
    dp_cost = std::min(dp_cost, kBigDouble);

    dp[i] = dp_cost;
  }

  logf(LOG_INFO, "DP Cost: %.2lf", dp[nblocks]);

  return 0;
}

double GetSumRange(std::vector<double> const& cum_sum, int a, int b) {
  if (a > 0) {
    return cum_sum[b] - cum_sum[a - 1];
  } else {
    return cum_sum[b];
  }
}

int AssignBlocksDP2(std::vector<double> const& costlist,
                    std::vector<int>& ranklist, int nranks) {
  double cost_total = std::accumulate(costlist.begin(), costlist.end(), 0.0);
  double cost_target = cost_total / nranks;
  logf(LOG_DBG2, "Target Cost: %.2lf", cost_target);

  std::vector<double> cum_costlist(costlist);
  int nblocks = costlist.size();
  int n_a = std::floor(nblocks * 1.0 / nranks);
  int n_b = std::ceil(nblocks * 1.0 / nranks);
  int nalloc_b = nblocks % nranks;
  int nalloc_a = nranks - nalloc_b;

  for (int i = 1; i < nblocks; i++) {
    cum_costlist[i] += cum_costlist[i - 1];
  }

  const double kBigDouble = cost_total * 1e3;
  //  std::vector<double> dp(nblocks + 1, kBigDouble);
  double dp[nalloc_a + 1][nalloc_b + 1];
  for (int i = 0; i <= nalloc_a; i++) {
    for (int j = 0; j <= nalloc_b; j++) {
      dp[i][j] = kBigDouble;
    }
  }

  dp[0][0] = 0;
  for (int i = 0; i <= nalloc_a; i++) {
    for (int j = 0; j <= nalloc_b; j++) {
      int l = i * n_a + j * n_b;

      // option 1. we add an n_a chunk ending at l - 1
      // chunk range: [l - n_a, l - 1]
      double opt1_cost = kBigDouble;
      if (l >= n_a and i > 0) {
        double opt1_cost_a = GetSumRange(cum_costlist, l - n_a, l - 1);
        double opt1_cost_b = dp[i - 1][j];
        opt1_cost = std::max(opt1_cost_a, opt1_cost_b);
      }

      // option 2. we add an n_b chunk ending at l - 1
      // chunk range: [l - n_b, l - 1]
      double opt2_cost = kBigDouble;
      if (l >= n_b and j > 0) {
        double opt2_cost_a = GetSumRange(cum_costlist, l - n_b, l - 1);
        double opt2_cost_b = dp[i][j - 1];
        opt2_cost = std::max(opt2_cost_a, opt2_cost_b);
      }

      double dp_cost = std::min(opt1_cost, opt2_cost);
      dp_cost = std::min(dp_cost, kBigDouble);

      if (dp_cost != kBigDouble) {
        dp[i][j] = dp_cost;
        logf(LOG_DBG2, "DP[%d][%d] = %.2lf", i, j, dp_cost);
      }
    }
  }

  logf(LOG_INFO, "DP Cost: %.2lf", dp[nalloc_a][nalloc_b]);

  int i = nalloc_a;
  int j = nalloc_b;
  int cur_rank = nranks - 1;
  while (i > 0 or j > 0) {
    int l = i * n_a + j * n_b;

    // option 1. we add an n_a chunk ending at l - 1
    // chunk range: [l - n_a, l - 1]
    double opt1_cost = kBigDouble;
    if (l >= n_a and i > 0) {
      double opt1_cost_a = GetSumRange(cum_costlist, l - n_a, l - 1);
      double opt1_cost_b = dp[i - 1][j];
      opt1_cost = std::max(opt1_cost_a, opt1_cost_b);
    }

    // option 2. we add an n_b chunk ending at l - 1
    // chunk range: [l - n_b, l - 1]
    double opt2_cost = kBigDouble;
    if (l >= n_b and j > 0) {
      double opt2_cost_a = GetSumRange(cum_costlist, l - n_b, l - 1);
      double opt2_cost_b = dp[i][j - 1];
      opt2_cost = std::max(opt2_cost_a, opt2_cost_b);
    }

    double dp_cost = std::min(opt1_cost, opt2_cost);
    // should not encounter invalid solutions while backtracking
    assert(dp_cost != kBigDouble);
    if (dp_cost == opt1_cost) {
      logf(LOG_DBG2, "Backtracking [%d][%d]->[%d][%d]", i, j, i - 1, j);
      MarkRange(ranklist, l - n_a, l - 1, cur_rank);
      i--;
    } else {
      logf(LOG_DBG2, "Backtracking [%d][%d]->[%d][%d]", i, j, i, j - 1);
      MarkRange(ranklist, l - n_b, l - 1, cur_rank);
      j--;
    }

    cur_rank--;
  }

  return 0;
}

int AssignBlocksContigImproved(std::vector<double> const& costlist,
                               std::vector<int>& ranklist, int nranks) {
  return AssignBlocksDP2(costlist, ranklist, nranks);
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
    if (pq.empty()) break;

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
  int nblocks = costlist.size();
  if (nblocks % nranks == 0) {
    logf(LOG_DBUG,
         "Blocks evenly divisible by nranks, using AssignBlocksContiguous");
    return AssignBlocksContiguous(costlist, ranklist, nranks);
  } else {
    return ::AssignBlocksContigImproved(costlist, ranklist, nranks);
  }
}
}  // namespace amr
