//
// Created by Ankush J on 4/10/23.
//

#include "policy_exec_ctx.h"

#include "lb_policies.h"

namespace amr {
int amr::PolicyExecutionContext::ExecuteTimestep(
    int nranks, const std::vector<double>& cost_alloc,
    const std::vector<double>& cost_actual) {
  int rv = 0;

  int nblocks = cost_alloc.size();
  assert(nblocks == cost_actual.size());

  std::vector<int> rank_list(nblocks, -1);
  std::vector<double> rank_times(nranks, 0);

  uint64_t ts_assign_beg = env_->NowMicros();
  rv = LoadBalancePolicies::AssignBlocksInternal(policy_, cost_alloc, rank_list,
                                                 nranks);
  uint64_t ts_assign_end = env_->NowMicros();

  if (rv) return rv;

  exec_time_us_ += (ts_assign_end - ts_assign_beg);

  for (int bid = 0; bid < nblocks; bid++) {
    int block_rank = rank_list[bid];
    rank_times[block_rank] += cost_actual[bid];
  }

  int const& (*max_func)(int const&, int const&) = std::max<int>;
  int rtmax = std::accumulate(rank_times.begin(), rank_times.end(),
                              rank_times.front(), max_func);
  uint64_t rtsum = std::accumulate(rank_times.begin(), rank_times.end(), 0ull);
  double rtavg = rtsum * 1.0 / nranks;

  excess_cost_ += (rtmax - rtavg);
  total_cost_avg_ += rtavg;
  total_cost_max_ += rtmax;

  ts_++;

  return rv;
}
}  // namespace amr