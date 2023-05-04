//
// Created by Ankush J on 4/10/23.
//

#pragma once

#include "common.h"
#include "lb_policies.h"
#include "lb_trigger.h"
#include "policy.h"
#include "policy_stats.h"
#include "utils.h"

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
                      std::vector<int>& refs, std::vector<int>& derefs);

  static void LogHeader(fort::char_table& table) {
    table << fort::header << "LoadBalancingPolicy"
          << "Timesteps";
    PolicyStats::LogHeader(table);
    table << "ExecTime" << fort::endr;
  }

  void LogSummary(fort::char_table& table) {
    table << opts_.policy_name
          << std::to_string(ts_succeeded_) + "/" + std::to_string(ts_invoked_);
    stats_.LogSummary(table);
    table << PolicyStats::FormatProp(exec_time_us_ / 1e6, "s") << fort::endr;
  }

  std::string Name() const { return opts_.policy_name; }

 private:
  void Bootstrap();

  static bool ComputeLBTrigger(TriggerPolicy tp, LoadBalanceState& state) {
    if (tp == TriggerPolicy::kEveryTimestep) return true;

    return (!state.refs.empty() || !state.derefs.empty());
  }

  static void ComputeCostsForLB(CostEstimationPolicy cep,
                                LoadBalanceState& state,
                                std::vector<double> const& costlist_oracle,
                                std::vector<double>& costlist_new) {
    int nblocks_cur = GetNumBlocksNext(state);

    switch (cep) {
      case CostEstimationPolicy::kOracleCost:
        costlist_new = costlist_oracle;
        break;
      case CostEstimationPolicy::kUnitCost:
        costlist_new = std::vector<double>(nblocks_cur, 1.0);
        break;
      case CostEstimationPolicy::kExtrapolatedCost:
        // If both refs and derefs are empty, these should be the same
        Utils::ExtrapolateCosts(state.costlist_prev, state.refs, state.derefs,
                                costlist_new);
        break;
      default:
        ABORT("Not implemented!");
    }

    assert(costlist_new.size() == nblocks_cur);
  }

  static int GetNumBlocksNext(LoadBalanceState const& state) {
    int nblocks = state.costlist_prev.size();
    assert(state.ranklist.size() == nblocks);
    int nblocks_next =
        nblocks + (state.refs.size() * 7) - (state.derefs.size() * 7 / 8);
    return nblocks_next;
  }

  int TriggerLB(const std::vector<double>& costlist);

  void EnsureOutputFile();

  static std::string GetLogPath(const char* output_dir,
                                const char* policy_name);

  const PolicyExecOpts opts_;
  LoadBalanceState lb_state_;
  PolicyStats stats_;

  pdlfs::WritableFile* fd_;
  int ts_invoked_;
  int ts_succeeded_;
  double exec_time_us_;

  friend class MiscTest;
};
}  // namespace amr
