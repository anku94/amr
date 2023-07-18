//
// Created by Ankush J on 7/13/23.
//

#pragma once

#include "inputs.h"
#include "scale_exec_ctx.h"
#include "trace_utils.h"

#include <utility>

namespace amr {
struct ScaleSimOpts {
  std::string output_dir;
  pdlfs::Env* env;
  int nblocks_beg;
  int nblocks_end;
};

struct RunProfile {
  int nranks;
  int nblocks;
};

class ScaleSim {
 public:
  explicit ScaleSim(ScaleSimOpts opts) : options_(std::move(opts)) {}

  void SetupAllPolicies() {
    PolicyExecOpts policy_opts;
    policy_opts.output_dir = options_.output_dir.c_str();
    policy_opts.env = options_.env;
    policy_opts.nranks = -1;
    policy_opts.nblocks_init = -1;

    policy_opts.SetPolicy("Contiguous/Unit-Cost",
                          LoadBalancePolicy::kPolicyContiguousActualCost,
                          CostEstimationPolicy::kUnitCost);
    policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy("LPT/Actual-Cost", LoadBalancePolicy::kPolicyLPT,
                          CostEstimationPolicy::kUnitCost);
    policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy("LPT/Actual-Cost", LoadBalancePolicy::kPolicyLPT,
                          CostEstimationPolicy::kUnitCost);
    policies_.emplace_back(policy_opts);
  }

  void Run() {
    logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());
    Utils::EnsureDir(options_.env, options_.output_dir);
    SetupAllPolicies();

    fort::char_table table;
    ScaleExecCtx::LogHeader(table);

    std::vector<RunProfile> run_profiles;
    GenRunProfiles(run_profiles, options_.nblocks_beg, options_.nblocks_end);
    std::vector<double> costs;

    for (auto& policy : policies_) {
      for (auto& r : run_profiles) {
        logf(LOG_DBUG, "[Running profile] nranks: %d, nblocks: %d", r.nranks,
             r.nblocks);

        if (costs.size() != r.nblocks) {
          costs.resize(r.nblocks);
          Inputs::GenerateCosts(costs);
        }

        policy.AssignBlocks(r.nranks, costs, table);
      }
    }
  }

 private:
  static void GenRunProfiles(std::vector<RunProfile>& v, int nb_beg,
                             int nb_end) {
    v.clear();

    for (int nblocks = nb_beg; nblocks <= nb_end; nblocks *= 2) {
      for (int nranks = nblocks; nranks <= nblocks; nranks *= 2) {
        v.emplace_back(RunProfile{nranks, nblocks});
      }
    }
  }

  ScaleSimOpts const options_;
  std::vector<ScaleExecCtx> policies_;
};
}  // namespace amr