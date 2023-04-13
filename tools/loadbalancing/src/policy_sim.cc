//
// Created by Ankush J on 4/10/23.
//

#include "policy_sim.h"

#include "policy.h"
#include "policy_exec_ctx.h"

namespace amr {
void PolicySim::LogSummary() {
  logf(LOG_INFO, "\n\nFinished trace replay. Summary:\n");
  for (auto& policy : policies_) policy.ctx.LogSummary();
}

void PolicySim::EnsureOutputDir() {
    pdlfs::Status s = env_->CreateDir(options_.output_dir.c_str());
    if (s.ok()) {
      logf(LOG_INFO, "\t- Created successfully.");
    } else if (s.IsAlreadyExists()) {
      logf(LOG_INFO, "\t- Already exists.");
    } else {
      logf(LOG_ERRO, "Failed to create output directory: %s (Reason: %s)",
           options_.output_dir, s.ToString().c_str());
    }
}

void PolicySim::InitializePolicies() {
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
}

void PolicySim::SimulateTrace() {
  logf(LOG_INFO, "[SimulateTrace] Looking for trace files in: \n\t%s",
       options_.prof_dir.c_str());

  std::vector<std::string> files = LocateRelevantFiles(options_.prof_dir);

  if (files.empty()) {
    ABORT("no trace files found!");
  }

  ProfSetReader psr;
  for (auto& f : files) {
    std::string full_path = std::string(options_.prof_dir) + "/" + f;
    logf(LOG_INFO, "[ProfSetReader] Adding trace file: %s", full_path.c_str());
    psr.AddProfile(full_path);
  }

  nts_ = 0;
  bad_ts_ = 0;

  std::vector<int> block_times;
  while (psr.ReadTimestep(block_times) > 0) {
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

std::vector<std::string> PolicySim::LocateRelevantFiles(
    const std::string& root_dir) {
  std::vector<std::string> files;
  env_->GetChildren(root_dir.c_str(), &files);

  logf(LOG_DBG2, "Enumerating directory: %s", root_dir.c_str());
  for (auto& f : files) {
    logf(LOG_DBG2, "- File: %s", f.c_str());
  }

  std::vector<std::string> regex_patterns = {
      R"(prof\.merged\.evt\d+\.csv)",
      R"(prof\.merged\.evt\d+\.mini\.csv)",
      R"(prof\.aggr\.evt\d+\.csv)",
  };

  for (auto& pattern : regex_patterns) {
    logf(LOG_DBG2, "Searching by pattern: %s", pattern.c_str());
    std::vector<std::string> relevant_files = FilterByRegex(files, pattern);

    for (auto& f : relevant_files) {
      logf(LOG_DBG2, "- Match: %s", f.c_str());
    }

    if (!relevant_files.empty()) return relevant_files;
  }

  return {};
}
}  // namespace amr
