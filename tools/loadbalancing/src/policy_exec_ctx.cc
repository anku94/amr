//
// Created by Ankush J on 5/4/23.
//

#include "policy_exec_ctx.h"
#include "policy_wopts.h"
#include "policy.h"

namespace amr {

PolicyExecCtx::PolicyExecCtx(PolicyExecOpts& opts)
    : opts_(opts),
      policy_(PolicyUtils::GetPolicy(opts_.policy_id)),
      use_cost_cache_(opts_.cost_policy ==
                      CostEstimationPolicy::kCachedExtrapolatedCost),
      ts_(0),
      ts_lb_invoked_(0),
      ts_lb_succeeded_(0),
      ts_since_last_lb_(0),
      cost_cache_(opts_.cache_ttl) {
  Bootstrap();
}

void PolicyExecCtx::Bootstrap() {
  assert(opts_.nblocks_init % opts_.nranks == 0);
  int nblocks_per_rank = opts_.nblocks_init / opts_.nranks;

  lb_state_.ranklist.clear();

  for (int i = 0; i < opts_.nranks; ++i) {
    for (int j = 0; j < nblocks_per_rank; ++j) {
      lb_state_.ranklist.push_back(i);
    }
  }

  lb_state_.costlist_prev = std::vector<double>(opts_.nblocks_init, 1.0);

  assert(lb_state_.ranklist.size() == opts_.nblocks_init);
  logf(LOG_DBG2, "[PolicyExecCtx] Bootstrapping. Num Blocks: %d, Ranklist: %zu",
       opts_.nblocks_init, lb_state_.ranklist.size());
}

int PolicyExecCtx::ExecuteTimestep(std::vector<double> const& costlist_oracle,
                                   std::vector<int> const& ranklist_actual,
                                   std::vector<int>& refs,
                                   std::vector<int>& derefs, double& exec_time) {
  int rv = 0;

  assert(lb_state_.costlist_prev.size() == lb_state_.ranklist.size());

  bool trigger_lb = ComputeLBTrigger(opts_.trigger_policy, lb_state_);
  if (trigger_lb) {
    std::vector<double> costlist_lb;
    ComputeCosts(ts_, costlist_oracle, costlist_lb);
    rv = TriggerLB(costlist_lb, exec_time);
    if (rv) {
      logf(LOG_WARN, "[PolicyExecCtx] TriggerLB failed!");
      ts_++;
      return rv;
    }
  }

  // Timestep is always evaluated using the oracle cost
  if (policy_.policy == LoadBalancePolicy::kPolicyActual) {
    assert(ranklist_actual.size() == costlist_oracle.size());
    logf(LOG_DBUG, "Logging with ranklist_actual (%zu)",
         ranklist_actual.size());
  } else {
    assert(lb_state_.ranklist.size() == costlist_oracle.size());
    logf(LOG_DBUG, "Logging with lb.ranklist (%zu)", lb_state_.ranklist.size());
  }

  lb_state_.costlist_prev = costlist_oracle;
  lb_state_.refs = refs;
  lb_state_.derefs = derefs;

  ts_++;
  return rv;
}

int PolicyExecCtx::TriggerLB(const std::vector<double>& costlist, double& exec_time) {
  int rv;
  std::vector<int> ranklist_lb;

  ts_lb_invoked_++;

  uint64_t lb_beg = pdlfs::Env::NowMicros();
  rv = LoadBalancePolicies::AssignBlocks(opts_.policy_id, costlist, ranklist_lb,
                                         opts_.nranks);
  uint64_t lb_end = pdlfs::Env::NowMicros();

  if (rv) return rv;

  ts_lb_succeeded_++;
  exec_time = (lb_end - lb_beg);

  lb_state_.ranklist = ranklist_lb;

  assert(lb_state_.ranklist.size() == costlist.size());

  return rv;
}
};  // namespace amr
