//
// Created by Ankush J on 4/28/23.
//

#pragma once

#include "bin_readers.h"
#include "common.h"
#include "policy.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"
#include "trace_utils.h"

#include <vector>

namespace amr {
struct BlockSimulatorOpts {
  int nblocks;
  int nranks;
  int nts;
  std::string prof_dir;
  std::string output_dir;
  pdlfs::Env* env;
  std::vector<int> events;
  const char* prof_time_combine_policy;
};

#define FAIL_IF(cond, msg) \
  if (cond) {              \
    logf(LOG_ERRO, msg);   \
    ABORT(msg);            \
  }

class BlockSimulator {
 public:
  explicit BlockSimulator(BlockSimulatorOpts& opts)
      : options_(opts),
        ref_reader_(options_.prof_dir),
        assign_reader_(options_.prof_dir),
        prof_reader_(Utils::LocateTraceFiles(options_.env, options_.prof_dir,
                                             options_.events),
                     Utils::ParseProfTimeCombinePolicy(
                         options_.prof_time_combine_policy)),
        nblocks_next_expected_(-1),
        num_lb_(0) {}

  void SetupAllPolicies() {
    policies_.clear();

    PolicyExecOpts policy_opts;
    policy_opts.output_dir = options_.output_dir.c_str();
    policy_opts.env = options_.env;
    policy_opts.nranks = options_.nranks;
    policy_opts.nblocks_init = options_.nblocks;
    // XXX: hardcoded for now
    policy_opts.trigger_interval = 1000;
    logf(LOG_INFO, "Hardcoded trigger interval: %d\n",
         policy_opts.trigger_interval);

    policy_opts.SetPolicy(
        "Actual/Actual-Cost", LoadBalancePolicy::kPolicyActual,
        CostEstimationPolicy::kExtrapolatedCost, TriggerPolicy::kEveryNTimesteps);
    policies_.emplace_back(policy_opts);

    //    policy_opts.SetPolicy(
    //        "Contiguous/Unit-Cost",
    //        LoadBalancePolicy::kPolicyContiguousActualCost,
    //        CostEstimationPolicy::kUnitCost, TriggerPolicy::kOnMeshChange);
    //    policies_.emplace_back(policy_opts);

    // policy_opts.SetPolicy("Contiguous/Unit-Cost-Alt",
    // LoadBalancePolicy::kPolicyContiguousUnitCost,
    // CostEstimationPolicy::kExtrapolatedCost,
    // TriggerPolicy::kOnMeshChange);
    // policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy("Contiguous/Extrapolated-Cost",
                          LoadBalancePolicy::kPolicyContiguousActualCost,
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    policies_.emplace_back(policy_opts);

    // policy_opts.SetPolicy(
    // "RoundRobin/Extrapolated-Cost", LoadBalancePolicy::kPolicyRoundRobin,
    // CostEstimationPolicy::kExtrapolatedCost, TriggerPolicy::kOnMeshChange);
    // policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy(
        "LPT/Extrapolated-Cost", LoadBalancePolicy::kPolicyLPT,
        CostEstimationPolicy::kExtrapolatedCost, TriggerPolicy::kEveryNTimesteps);
    policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy("kContigImproved/Extrapolated-Cost",
                          LoadBalancePolicy::kPolicyContigImproved,
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy(
        "CppIter/Extrapolated-Cost", LoadBalancePolicy::kPolicyCppIter,
        CostEstimationPolicy::kExtrapolatedCost, TriggerPolicy::kEveryNTimesteps);
    policies_.emplace_back(policy_opts);

    // PolicyOptsILP ilp_opts;
    // ilp_opts.obj_lb_time_limit = 100;
    // ilp_opts.obj_loc_time_limit = 100;

    // ilp_opts.obj_lb_rel_tol = 0.10;
    // ilp_opts.obj_lb_mip_gap = 0.10;
    // ilp_opts.obj_loc_mip_gap = 0.10;
    // policy_opts.SetPolicy("ILP_10PCT/Actual-Cost",
                          // LoadBalancePolicy::kPolicyILP,
                          // CostEstimationPolicy::kCachedExtrapolatedCost,
                          // TriggerPolicy::kEveryNTimesteps);
    // policy_opts.SetLBOpts(ilp_opts);

    // policy_opts.cache_ttl = 400;
    // policy_opts.SetPolicy("LPT/Extrapolated-Cost-Cached",
                          // LoadBalancePolicy::kPolicyLPT,
                          // CostEstimationPolicy::kCachedExtrapolatedCost,
                          // TriggerPolicy::kEveryNTimesteps);
    // policies_.emplace_back(policy_opts);
    //
    //    policy_opts.SetPolicy(
    //        "LPT/Actual-Cost-Oracle", LoadBalancePolicy::kPolicyLPT,
    //        CostEstimationPolicy::kOracleCost, TriggerPolicy::kOnMeshChange);
    //    policies_.emplace_back(policy_opts);
    //
    //    policy_opts.SetPolicy(
    //        "LPT/Extrapolated-Cost-EveryTS", LoadBalancePolicy::kPolicyLPT,
    //        CostEstimationPolicy::kExtrapolatedCost,
    //        TriggerPolicy::kEveryTimestep);
    //    policies_.emplace_back(policy_opts);
    //
    //    policy_opts.SetPolicy(
    //        "LPT/Actual-Cost-Oracle-EveryTS", LoadBalancePolicy::kPolicyLPT,
    //        CostEstimationPolicy::kOracleCost, TriggerPolicy::kEveryTimestep);
    //    policies_.emplace_back(policy_opts);
  }

  int InvokePolicies(std::vector<double> const& cost_oracle,
                     std::vector<int>& ranklist_actual, std::vector<int>& refs,
                     std::vector<int>& derefs) {
    int rv = 0;

    for (auto& policy : policies_) {
      rv = policy.ExecuteTimestep(cost_oracle, ranklist_actual, refs, derefs);
    }

    return rv;
  }

  void Run();

  int RunTimestep(int& ts, int sub_ts);

 private:
  int ReadTimestepInternal(int ts, int sub_ts, std::vector<int>& refs,
                           std::vector<int>& derefs,
                           std::vector<int>& assignments,
                           std::vector<int>& times);

  void LogSummary(fort::char_table& table);

  BlockSimulatorOpts const options_;

  RefinementReader ref_reader_;
  AssignmentReader assign_reader_;
  ProfSetReader prof_reader_;

  std::vector<int> ranklist_;
  int nblocks_next_expected_;
  int num_lb_;

  std::vector<PolicyExecCtx> policies_;
};
}  // namespace amr
