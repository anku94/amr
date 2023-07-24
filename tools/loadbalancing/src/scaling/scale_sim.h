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
  explicit ScaleSim(ScaleSimOpts opts)
      : options_(std::move(opts)),
        log_(options_.env, options_.output_dir + "/scalesim.log.csv") {}

  void SetupAllPolicies() {
    PolicyExecOpts policy_opts;
    policy_opts.output_dir = options_.output_dir.c_str();
    policy_opts.env = options_.env;
    policy_opts.nranks = -1;
    policy_opts.nblocks_init = -1;

    //    policy_opts.SetPolicy("Contiguous/Unit-Cost",
    //                          LoadBalancePolicy::kPolicyContiguousActualCost,
    //                          CostEstimationPolicy::kUnitCost);
    //    policies_.emplace_back(policy_opts);
    //
    policy_opts.SetPolicy("LPT/Actual-Cost", LoadBalancePolicy::kPolicyLPT,
                          CostEstimationPolicy::kUnitCost);
    policies_.emplace_back(policy_opts);

    policy_opts.SetPolicy("CPP/Actual-Cost",
                          LoadBalancePolicy::kPolicyContigImproved,
                          CostEstimationPolicy::kUnitCost);
    policies_.emplace_back(policy_opts);
    //
    //    policy_opts.SetPolicy("CPP-Iter/Actual-Cost",
    //                          LoadBalancePolicy::kPolicyCppIter,
    //                          CostEstimationPolicy::kUnitCost);
    //    policies_.emplace_back(policy_opts);

    PolicyOptsILP ilp_opts;
    ilp_opts.mip_gap = 0.05;
    ilp_opts.obj_lb_rel_gap = 0.05;
    policy_opts.SetPolicy("ILP_5PCT/Actual-Cost", LoadBalancePolicy::kPolicyILP,
                          CostEstimationPolicy::kUnitCost);
    policy_opts.SetLBOpts(ilp_opts);
    policies_.emplace_back(policy_opts);

    ilp_opts.mip_gap = 0.10;
    ilp_opts.obj_lb_rel_gap = 0.10;
    policy_opts.SetPolicy("ILP_10PCT/Actual-Cost",
                          LoadBalancePolicy::kPolicyILP,
                          CostEstimationPolicy::kUnitCost);
    policy_opts.SetLBOpts(ilp_opts);
    policies_.emplace_back(policy_opts);

    ilp_opts.mip_gap = 0.20;
    ilp_opts.obj_lb_rel_gap = 0.20;
    policy_opts.SetPolicy("ILP_20PCT/Actual-Cost",
                          LoadBalancePolicy::kPolicyILP,
                          CostEstimationPolicy::kUnitCost);
    policy_opts.SetLBOpts(ilp_opts);
    policies_.emplace_back(policy_opts);
  }

  void Run() {
    logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());
    Utils::EnsureDir(options_.env, options_.output_dir);
    SetupAllPolicies();

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

        policy.AssignBlocks(r.nranks, costs, log_);
      }
    }

    logf(LOG_INFO, "\n%s", log_.GetTabularStr().c_str());
  }

 private:
  static void GenRunProfiles(std::vector<RunProfile>& v, int nb_beg,
                             int nb_end) {
    v.clear();

    for (int nblocks = nb_beg; nblocks <= nb_end; nblocks *= 2) {
      for (int nranks = nblocks / 5; nranks <= nblocks / 5; nranks *= 2) {
        v.emplace_back(RunProfile{nranks, nblocks});
      }
    }
  }

  ScaleSimOpts const options_;
  std::vector<ScaleExecCtx> policies_;
  ScaleExecLog log_;
};
}  // namespace amr