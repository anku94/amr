//
// Created by Ankush J on 7/14/23.
//

#pragma once

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
    for (int iter = 0; iter < kMaxIters; iter++) {
      rv = LoadBalancePolicies::AssignBlocks(opts_.lb_policy, cost_list,
                                             rank_list, nranks);
    }
    uint64_t lb_end = pdlfs::Env::NowMicros();

    double rtavg, rtmax;
    PolicyUtils::ComputePolicyCosts(nranks, cost_list, rank_list, rank_times,
                                    rtavg, rtmax);

    double iter_time = (lb_end - lb_beg) / kMaxIters;

    log.WriteRow(opts_.policy_name, cost_list.size(), nranks, iter_time, rtavg,
                 rtmax);

    return rv;
  }

  static void LogHeader(fort::char_table& table) {
    table << fort::header << "Policy"
          << "Num Blocks"
          << "Num Ranks"
          << "Iter Time"
          << "Rank Time (avg)"
          << "Rank Time (max)" << fort::endr;
  }

 private:
  PolicyExecOpts const opts_;
  static constexpr int kMaxIters = 25;
};
}  // namespace amr