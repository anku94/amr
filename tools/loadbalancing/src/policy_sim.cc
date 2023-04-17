//
// Created by Ankush J on 4/10/23.
//

#include "policy_sim.h"

#include "fort.hpp"
#include "policy.h"
#include "policy_exec_ctx.h"

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

void PolicySim::EnsureOutputDir() {
  pdlfs::Status s = env_->CreateDir(options_.output_dir.c_str());
  if (s.ok()) {
    logf(LOG_INFO, "\t- Created successfully.");
  } else if (s.IsAlreadyExists()) {
    logf(LOG_INFO, "\t- Already exists.");
  } else {
    logf(LOG_ERRO, "Failed to create output directory: %s (Reason: %s)",
         options_.output_dir.c_str(), s.ToString().c_str());
  }
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

std::vector<std::string> PolicySim::LocateTraceFiles(
    const std::string& search_dir) const {
  logf(LOG_INFO, "[SimulateTrace] Looking for trace files in: \n\t%s",
       search_dir.c_str());

  std::vector<std::string> files;
  env_->GetChildren(search_dir.c_str(), &files);

  logf(LOG_DBG2, "Enumerating directory: %s", search_dir.c_str());
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

    if (!relevant_files.empty()) break;
  }

  if (files.empty()) {
    ABORT("no trace files found!");
  }

  std::vector<std::string> all_fpaths;

  for (auto& f : files) {
    std::string full_path = std::string(search_dir) + "/" + f;
    logf(LOG_INFO, "[ProfSetReader] Adding trace file: %s", full_path.c_str());
    all_fpaths.push_back(full_path);
  }

  return all_fpaths;
}

void PolicySim::SimulateTrace(int ts_beg, int ts_end) {
  ProfSetReader psr(LocateTraceFiles(options_.prof_dir));

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
