//
// Created by Ankush J on 7/14/23.
//

#pragma once

#include "constants.h"
#include "fort.hpp"
#include "lb_policies.h"
#include "outputs.h"
#include "policy.h"
#include "writable_file.h"

#include <pdlfs-common/env.h>

namespace amr {
class ScaleExecCtx {
 public:
  explicit ScaleExecCtx(const PolicyExecOpts& opts) : opts_(opts) {}

  int AssignBlocks(int nranks, std::vector<double> const& cost_list,
                   ScaleExecLog& log) {
    int rv;
    std::vector<int> rank_list;
    std::vector<double> rank_times;

    uint64_t lb_beg = pdlfs::Env::NowMicros();
    for (int iter = 0; iter < Constants::kScaleSimIters; iter++) {
      rv = LoadBalancePolicies::AssignBlocks(opts_.lb_policy, cost_list,
                                             rank_list, nranks, opts_.GetLBOpts());
    }
    uint64_t lb_end = pdlfs::Env::NowMicros();

    double rtavg, rtmax;
    PolicyUtils::ComputePolicyCosts(nranks, cost_list, rank_list, rank_times,
                                    rtavg, rtmax);
    double iter_time = (lb_end - lb_beg) / Constants::kScaleSimIters;
    double loc_cost = PolicyUtils::ComputeLocCost(rank_list) * 100;

    log.WriteRow(opts_.policy_name, cost_list.size(), nranks, iter_time, rtavg,
                 rtmax, loc_cost);

    return rv;
  }

 private:
  PolicyExecOpts const opts_;
};
}  // namespace amr