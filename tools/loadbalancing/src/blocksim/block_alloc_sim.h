//
// Created by Ankush J on 4/28/23.
//

#pragma once

#include "bin_readers.h"
#include "common.h"
#include "policy.h"
#include "policy_exec_ctx.h"
#include "policy_stats.h"
#include "prof_set_reader.h"
#include "trace_utils.h"

#include <vector>

namespace amr {
struct BlockSimulatorOpts {
  int nblocks;
  int nranks;
  int nts;
  int nts_toskip;
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

    policy_opts.SetPolicy("Actual/Actual-Cost", "actual",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("LPT/Extrapolated-Cost", "lpt",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("kContigImproved/Extrapolated-Cost", "cdp",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("CppIter/Extrapolated-Cost", "cdpi50",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid10",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid20",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid30",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid50",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid70",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

    policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid90",
                          CostEstimationPolicy::kExtrapolatedCost,
                          TriggerPolicy::kEveryNTimesteps);
    SetupPolicy(policy_opts);

  }

  void SetupPolicy(PolicyExecOpts& opts) {
    policies_.emplace_back(opts);
    stats_.emplace_back(opts);
  }

  int InvokePolicies(std::vector<double> const& cost_oracle,
                     std::vector<int>& ranklist_actual, std::vector<int>& refs,
                     std::vector<int>& derefs) {
    int rv = 0;

    int npolicies = policies_.size();

    for (int pidx = 0; pidx < npolicies; ++pidx) {
      auto& policy = policies_[pidx];
      double exec_time = 0;
      rv = policy.ExecuteTimestep(cost_oracle, ranklist_actual, refs, derefs, exec_time);
      if (rv != 0) break;

      if (policy.IsActualPolicy()) {
        stats_[pidx].LogTimestep(cost_oracle, ranklist_actual, exec_time);
      } else {
        stats_[pidx].LogTimestep(cost_oracle, policy.GetRanklist(), exec_time);
      }
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
  std::vector<PolicyStats> stats_;
};
}  // namespace amr
