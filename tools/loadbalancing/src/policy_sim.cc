//
// Created by Ankush J on 4/10/23.
//

#include "policy_sim.h"

#include "fort.hpp"
#include "policy.h"
#include "policy_exec_ctx.h"
#include "utils.h"

namespace amr {
void PolicySim::LogSummary() {
  logf(LOG_INFO, "\n\nFinished trace replay. Summary:\n");
  for (auto& policy : policies_) policy.ctx.LogSummary();

  logf(LOG_INFO, "-------------------");
  logf(LOG_INFO, "Bad TS Count: %d/%d", bad_ts_, nts_);
  logf(LOG_INFO, "Run Finished.");
}

void PolicySim::LogSummary(fort::char_table& table) {
  table << fort::header << "Policy"
        << "Timesteps"
        << "ExcessCost"
        << "AvgCost"
        << "MaxCost"
        << "LocScore"
        << "ExecTime" << fort::endr;

  for (auto& policy : policies_) {
    policy.ctx.LogSummary(table);
  }

  logf(LOG_INFO, "\n%s", table.to_string().c_str());
}

void PolicySim::SetupAll() {
  policies_.emplace_back("Contiguous/Unit-Cost", Policy::kPolicyContiguous,
                         true, options_);
  policies_.emplace_back("Contiguous/Actual-Cost", Policy::kPolicyContiguous,
                         false, options_);
  policies_.emplace_back("RoundRobin/Actual-Cost", Policy::kPolicyRoundRobin,
                         false, options_);
  policies_.emplace_back("SPT/Actual-Cost", Policy::kPolicySPT, false,
                         options_);
  policies_.emplace_back("LPT/Actual-Cost", Policy::kPolicyLPT, false,
                         options_);
  // Too slow to run in the "ALL" mode
  //  policies_.emplace_back("ILP/Actual-Cost", Policy::kPolicyILP, false,
  //                         options_);
}

void PolicySim::SimulateTrace(int ts_beg, int ts_end) {
  ProfSetReader psr(Utils::LocateTraceFiles(options_.env, options_.prof_dir));

  nts_ = 0;
  bad_ts_ = 0;

  std::vector<int> block_times;
  while (psr.ReadTimestep(block_times) > 0) {
    if (nts_ < ts_beg) {
      nts_++;
      continue;
    } else if (nts_ >= ts_end) {
      break;
    }

    int rv = InvokePolicies(block_times);
    if (rv) {
      logf(LOG_WARN, "\n ====> !!!! TS %d seems bad !!!!", nts_);
      bad_ts_++;
    }

    nts_++;
    if (nts_ % 100 == 0) {
      printf("\rTS Read: %d", nts_);
      // break;
    }
  }
}

int PolicySim::InvokePolicies(std::vector<int>& cost_actual) {
  std::vector<double> cost_naive(cost_actual.size(), 1.0f);
  std::vector<double> cost_actual_lf(cost_actual.begin(), cost_actual.end());

  for (int pidx = 0; pidx < policies_.size(); pidx++) {
    int rv;
    auto& policy = policies_[pidx];

    if (policy.unit_cost) {
      rv = policy.ctx.ExecuteTimestep(cost_naive, cost_actual_lf);
    } else {
      rv = policy.ctx.ExecuteTimestep(cost_actual_lf, cost_actual_lf);
    }

    if (rv) return rv;
  }

  return 0;
}
}  // namespace amr
