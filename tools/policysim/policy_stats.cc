#include "policy_stats.h"

namespace amr {
void PolicyStats::LogTimestep(std::vector<double> const& cost_actual,
                              std::vector<int> const& rank_list, double exec_time_ts) {
  exec_time_us_ += exec_time_ts;

  auto nranks = opts_.nranks;

  std::vector<double> rank_times(nranks, 0);
  double rtavg, rtmax;

  PolicyUtils::ComputePolicyCosts(nranks, cost_actual, rank_list, rank_times,
                                  rtavg, rtmax);

  excess_cost_ += (rtmax - rtavg);
  total_cost_avg_ += rtavg;
  total_cost_max_ += rtmax;
  locality_score_sum_ += PolicyUtils::ComputeLocCost(rank_list);

  WriteSummary(fd_summ_, rtavg, rtmax);
  WriteDetailed(fd_det_, cost_actual, rank_list);
  WriteRankSums(fd_ranksum_, rank_times);

  ts_++;
}
}  // namespace amr
