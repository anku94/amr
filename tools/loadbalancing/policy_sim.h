#pragma once

#include "lb_policies.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"

#include "pdlfs-common/env.h"

#include <regex>
#include <vector>

namespace amr {
struct PolicySimOptions {
  pdlfs::Env* env;
  const char* prof_dir;
};

class PolicyExecutionContext;

class PolicySim {
 public:
  explicit PolicySim(const PolicySimOptions& options)
      : options_(options), env_(options.env), nts_(0), bad_ts_(0) {}

  void Run() {
    InitializePolicies();
    SimulateTrace();
    LogSummary();

    logf(LOG_INFO, "-------------------");
    logf(LOG_INFO, "Bad TS Count: %d/%d", bad_ts_, nts_);
    logf(LOG_INFO, "Run Finished.");
  }

  void InitializePolicies();

  void SimulateTrace();

  void LogSummary();

  int InvokePolicies(std::vector<int>& cost_actual);

 private:
  std::vector<std::string> LocateRelevantFiles(const std::string& root_dir);

  static std::vector<std::string> FilterByRegex(
      std::vector<std::string>& strings, std::string regex_pattern) {
    std::vector<std::string> matches;
    const std::regex regex_obj(regex_pattern);

    for (auto& s : strings) {
      std::smatch match_obj;
      if (std::regex_match(s, match_obj, regex_obj)) {
        matches.push_back(s);
      }
    }
    return matches;
  }

  const PolicySimOptions options_;
  pdlfs::Env* const env_;
  std::vector<PolicyExecutionContext> policies_;

  int nts_;
  int bad_ts_;
};
}  // namespace amr