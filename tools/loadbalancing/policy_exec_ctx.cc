//
// Created by Ankush J on 4/10/23.
//

#include "policy_exec_ctx.h"

#include "lb_policies.h"

//namespace amr {
//int amr::PolicyExecutionContext::ExecuteTimestep(
//    int nranks, const std::vector<double>& cost_alloc,
//    const std::vector<double>& cost_actual) {
//  int rv;
//  int nblocks = cost_alloc.size();
//  assert(nblocks == cost_actual.size());
//
//  std::vector<int> rank_list(nblocks, -1);
//
//  uint64_t ts_assign_beg = pdlfs::Env::NowMicros();
//  rv = LoadBalancePolicies::AssignBlocksInternal(policy_, cost_alloc, rank_list,
//                                                 nranks);
//  uint64_t ts_assign_end = pdlfs::Env::NowMicros();
//  if (rv) return rv;
//
//  stats_.LogTimestep(nranks, fd_, cost_actual, rank_list);
//  exec_time_us_ += (ts_assign_end - ts_assign_beg);
//  ts_++;
//  return rv;
//}
//}  // namespace amr