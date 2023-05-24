//
// Created by Ankush J on 4/10/23.
//

#include "policy_sim.h"

#include "fort.hpp"
#include "policy.h"
#include "policy_exec_ctx.h"
#include "trace_utils.h"

namespace amr {
void PolicySim::LogSummary(fort::char_table& table) {
  PolicyExecCtx::LogHeader(table);
  for (auto& policy : policies_) {
    policy.LogSummary(table);
  }

  logf(LOG_INFO, "\n%s", table.to_string().c_str());
}

void PolicySim::SetupAll() {
  ABORT("Deprecated. Use BlockSim instead");
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

int PolicySim::InvokePolicies(std::vector<int>& cost_oracle) {
  int rv;
  std::vector<double> cost_oracle_lf(cost_oracle.begin(), cost_oracle.end());
  std::vector<int> refs(0), derefs(0);

  for (auto& policy: policies_) {
    // PolicySim is no longer functional at all
//    rv = policy.ExecuteTimestep(cost_oracle_lf, <#initializer #>, refs, derefs,
//                                0);
    if (rv) return rv;
  }

  return 0;
}
}  // namespace amr
