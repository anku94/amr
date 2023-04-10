//
// Created by Ankush J on 4/10/23.
//

#include "policy_sim.h"

#include "policy_exec_ctx.h"

namespace amr {
void PolicySim::InitializePolicies() {
  policies_.emplace_back("Contiguous/Unit-Cost", Policy::kPolicyContiguous,
                         env_);
  policies_.emplace_back("Contiguous/Actual-Cost", Policy::kPolicyContiguous,
                         env_);
  policies_.emplace_back("RoundRobin/Actual-Cost", Policy::kPolicyRoundRobin,
                         env_);
  policies_.emplace_back("SPT/Actual-Cost", Policy::kPolicySPT, env_);
  policies_.emplace_back("LPT/Actual-Cost", Policy::kPolicyLPT, env_);
}

void PolicySim::SimulateTrace() {
  logf(LOG_INFO, "[SimulateTrace] Looking for trace files in: \n\t%s",
       options_.prof_dir);

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

  int nts = 0;
  std::vector<int> block_times;
  while (psr.ReadTimestep(block_times) > 0) {
    InvokePolicies(block_times);
    nts++;
    if (nts % 100 == 0) {
      printf("\rTS Read: %d", nts);
      // break;
    }
  }
}

void PolicySim::InvokePolicies(std::vector<int>& cost_actual) {
  int nranks = 512;

  std::vector<double> cost_naive(cost_actual.size(), 1.0f);
  std::vector<double> cost_actual_lf(cost_actual.begin(), cost_actual.end());

  policies_[0].ExecuteTimestep(nranks, cost_naive, cost_actual_lf);
  policies_[1].ExecuteTimestep(nranks, cost_actual_lf, cost_actual_lf);
  policies_[2].ExecuteTimestep(nranks, cost_actual_lf, cost_actual_lf);
  policies_[3].ExecuteTimestep(nranks, cost_actual_lf, cost_actual_lf);
  policies_[4].ExecuteTimestep(nranks, cost_actual_lf, cost_actual_lf);
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