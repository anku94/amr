#include "policy_stats.h"

#include "policy_wopts.h"

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

void PolicyStats::LogHeader(fort::char_table& table) {
  table << fort::header << "Name"
        << "LB Policy"
        << "Cost Policy"
        << "Trigger Policy"
        << "Timesteps"
        << "ExcessCost"
        << "AvgCost"
        << "MaxCost"
        << "LocScore"
        << "ExecTime" << fort::endr;
}

void PolicyStats::LogSummary(fort::char_table& table) const {
  auto policy = PolicyUtils::GetPolicy(opts_.policy_id);

  table << opts_.policy_id << policy.name
        << PolicyUtils::PolicyToString(opts_.cost_policy)
        << PolicyUtils::PolicyToString(opts_.trigger_policy);

  table << FormatProp(excess_cost_ / 1e6, "s")
        << FormatProp(total_cost_avg_ / 1e6, "s")
        << FormatProp(total_cost_max_ / 1e6, "s")
        << FormatProp(locality_score_sum_ * 100 / ts_, "%");

  table << FormatProp(exec_time_us_ / 1e6, "s") << fort::endr;
}

}  // namespace amr
