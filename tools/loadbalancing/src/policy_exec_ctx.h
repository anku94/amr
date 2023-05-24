//
// Created by Ankush J on 4/10/23.
//

#pragma once

#include "common.h"
#include "cost_cache.h"
#include "lb_policies.h"
#include "policy.h"
#include "policy_stats.h"
#include "trace_utils.h"

#include <pdlfs-common/env.h>
#include <regex>

namespace amr {

struct LoadBalanceState {
  std::vector<double> costlist_prev;
  std::vector<int> ranklist;
  std::vector<int> refs;
  std::vector<int> derefs;
};

class PolicyExecCtx {
 public:
  PolicyExecCtx(PolicyExecOpts& opts);

  // Safe move constructor for fd_
  PolicyExecCtx(PolicyExecCtx&& rhs) noexcept;

  PolicyExecCtx(const PolicyExecCtx& rhs) = delete;

  PolicyExecCtx& operator=(PolicyExecCtx&& rhs) = delete;

  ~PolicyExecCtx() {
    if (fd_) {
      pdlfs::Status s;
      SAFE_IO(fd_->Close(), "Close failed");
    }
  }

  int ExecuteTimestep(std::vector<double> const& costlist_oracle,
                      std::vector<int> const& ranklist_actual,
                      std::vector<int>& refs, std::vector<int>& derefs);

  static int GetNumBlocksNext(int nblocks, int nrefs, int nderefs) {
    int nblocks_next = nblocks + (nrefs * 7) - (nderefs * 7 / 8);
    return nblocks_next;
  }

  static void LogHeader(fort::char_table& table);

  void LogSummary(fort::char_table& table);

  std::string Name() const { return opts_.policy_name; }

 private:
  void Bootstrap();

  static bool ComputeLBTrigger(TriggerPolicy tp, LoadBalanceState& state) {
    if (tp == TriggerPolicy::kEveryTimestep) return true;

    return (!state.refs.empty() || !state.derefs.empty());
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
        PolicyUtils::ExtrapolateCosts(state.costlist_prev, state.refs,
                                      state.derefs, costlist_new);
        break;
      default:
        ABORT("Not implemented!");
    }
  }

  int TriggerLB(const std::vector<double>& costlist);

  void EnsureOutputFile();

  static std::string GetLogPath(const char* output_dir,
                                const char* policy_name);

  const PolicyExecOpts opts_;
  bool use_cost_cache_;

  LoadBalanceState lb_state_;
  PolicyStats stats_;
  CostCache cost_cache_;

  pdlfs::WritableFile* fd_;

  int ts_;
  int ts_lb_invoked_;
  int ts_lb_succeeded_;
  double exec_time_us_;

  friend class MiscTest;
};
}  // namespace amr
