//
// Created by Ankush J on 5/22/23.
//

#include <algorithm>
#include <cassert>
#include <vector>

#include "common.h"
#include "lb_policies.h"

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

  logv(__LOG_ARGS__, LOG_DBG2, "K: %d, Rolling Max: %.2lf, Rolling Min: %.2lf",
       k, rolling_max, rolling_min);
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
  logv(__LOG_ARGS__, LOG_DBG2, "MarkRange marking [%d, %d] with %d", start, end,
       flag);

  for (int i = start; i <= end; i++) {
    ranklist[i] = flag;
  }
  return true;
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
  logv(__LOG_ARGS__, LOG_DBG2, "Target Cost: %.2lf", cost_target);

  std::vector<double> cum_costlist(costlist);
  int nblocks = costlist.size();

  for (int i = 1; i < nblocks; i++) {
    cum_costlist[i] += cum_costlist[i - 1];
  }

  const double kBigDouble = cost_total * 1000;

  int n = nblocks, r = nranks;
  std::vector<std::vector<double>> dp(n + 1,
                                      std::vector<double>(r + 1, kBigDouble));

  for (int nr = 0; nr <= r; nr++) {
    dp[0][nr] = 0;
  }

  for (int ni = 1; ni <= n; ni++) {
    dp[ni][1] = GetSumRange(cum_costlist, 0, ni - 1);

    for (int ri = 1; ri <= r; ri++) {
      // lets' allocate 0 <= j <= ni blocks to ri
      for (int j = 0; j <= ni; j++) {
        double cost_j = GetSumRange(cum_costlist, ni - j, ni - 1);
        double cost_before_j = dp[ni - j][ri - 1];
        double cost_max = std::max(cost_j, cost_before_j);
        logv(__LOG_ARGS__, LOG_DBG3,
             "J: %d, CJ: %.2lf, CB: %.2lf, CM: %.2lf",
             j, cost_j, cost_before_j, cost_max);

        dp[ni][ri] = std::min(dp[ni][ri], cost_max);
      }

      logv(__LOG_ARGS__, LOG_DBG2, "DP[%d][%d]: %.2lf", ni, ri, dp[ni][ri]);
    }
  }

  logv(__LOG_ARGS__, LOG_DBG2, "Final DP[%d][%d]: %.2lf", n, r, dp[n][r]);

  return 0;
}
}  // namespace

namespace amr {
int LoadBalancePolicies::AssignBlocksContigImproved2(
    std::vector<double> const& costlist, std::vector<int>& ranklist,
    int nranks) {
  int nblocks = costlist.size();
  if (nblocks % nranks == 0) {
    logv(__LOG_ARGS__, LOG_DBUG,
         "Blocks evenly divisible by nranks_, using AssignBlocksContiguous");
    return AssignBlocksContiguous(costlist, ranklist, nranks);
  } else {
    return ::AssignBlocksDP2(costlist, ranklist, nranks);
  }
}
}  // namespace amr
