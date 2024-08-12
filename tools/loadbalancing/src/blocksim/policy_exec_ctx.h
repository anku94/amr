//
// Created by Ankush J on 4/10/23.
//

#pragma once

#include <pdlfs-common/env.h>

#include "common.h"
#include "cost_cache.h"
#include "policy_utils.h"
#include "policy_wopts.h"

namespace amr {

struct LoadBalanceState {
  std::vector<double> costlist_prev;
  std::vector<int> ranklist;
  std::vector<int> refs;
  std::vector<int> derefs;
};

class PolicyStats;

class PolicyExecCtx {
 public:
  PolicyExecCtx(PolicyExecOpts& opts);

  int ExecuteTimestep(std::vector<double> const& costlist_oracle,
                      std::vector<int> const& ranklist_actual,
                      std::vector<int>& refs, std::vector<int>& derefs,
                      double& exec_time);

  static int GetNumBlocksNext(int nblocks, int nrefs, int nderefs) {
    // int nblocks_next = nblocks + (nrefs * 7) - (nderefs * 7 / 8);
    int nblocks_next = nblocks + (nrefs * 3) - (nderefs * 3 / 4);
    return nblocks_next;
  }

  std::string Name() const { return policy_.name; }

  bool IsActualPolicy() const {
    return policy_.policy == LoadBalancePolicy::kPolicyActual;
  }

  std::vector<int> GetRanklist() const {
    if (policy_.policy == LoadBalancePolicy::kPolicyActual) {
      // raise a static error
      ABORT("Cannot get ranklist for actual policy");
    }

    return lb_state_.ranklist;
  }

  void GetTimestepCount(int& ts_succeeded, int& ts_invoked) const {
    ts_succeeded = ts_lb_succeeded_;
    ts_invoked = ts_lb_invoked_;
  }

 private:
  void Bootstrap();

  bool ComputeLBTrigger(TriggerPolicy tp, LoadBalanceState& state) {
    bool ref_trig = !state.refs.empty() || !state.derefs.empty();
    bool trig_intvl =
        (tp == TriggerPolicy::kEveryTimestep) ? 1 : opts_.trigger_interval;
    bool ts_trig = (ts_ % trig_intvl == 0);

    return ts_trig | ref_trig;
  }

  void ComputeCosts(int ts, std::vector<double> const& costlist_oracle,
                    std::vector<double>& costlist_new) {
    int nblocks_cur =
        GetNumBlocksNext(lb_state_.costlist_prev.size(), lb_state_.refs.size(),
                         lb_state_.derefs.size());

    if (use_cost_cache_ and cost_cache_.Get(ts, nblocks_cur, costlist_new)) {
      return;
    }

    ComputeCostsInternal(opts_.cost_policy, lb_state_, costlist_oracle,
                         costlist_new);

    if (use_cost_cache_) {
      cost_cache_.Put(ts, costlist_oracle);
    }

    assert(costlist_new.size() == nblocks_cur);
  }

  static void ComputeCostsInternal(CostEstimationPolicy cep,
                                   LoadBalanceState& state,
                                   std::vector<double> const& costlist_oracle,
                                   std::vector<double>& costlist_new) {
    int nblocks_cur = GetNumBlocksNext(state.costlist_prev.size(),
                                       state.refs.size(), state.derefs.size());

    switch (cep) {
      case CostEstimationPolicy::kOracleCost:
        costlist_new = costlist_oracle;
        break;
      case CostEstimationPolicy::kUnitCost:
        costlist_new = std::vector<double>(nblocks_cur, 1.0);
        break;
      case CostEstimationPolicy::kExtrapolatedCost:
      case CostEstimationPolicy::kCachedExtrapolatedCost:
        // If both refs and derefs are empty, these should be the same
        PolicyUtils::ExtrapolateCosts2D(state.costlist_prev, state.refs,
                                        state.derefs, costlist_new);
        break;
      default:
        ABORT("Not implemented!");
    }
  }

  int TriggerLB(const std::vector<double>& costlist, double& exec_time);

  const PolicyExecOpts opts_;
  const LBPolicyWithOpts policy_;
  bool use_cost_cache_;

  LoadBalanceState lb_state_;
  CostCache cost_cache_;

  int ts_;
  int ts_lb_invoked_;
  int ts_lb_succeeded_;
  int ts_since_last_lb_;

  friend class MiscTest;
  friend class PolicyStats;
};
}  // namespace amr
