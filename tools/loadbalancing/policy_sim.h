#pragma once

#include "lb_policies.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"

#include "pdlfs-common/env.h"

#include <regex>
#include <vector>

namespace amr {
enum class Policy {
  kPolicyContiguous,
  kPolicyRoundRobin,
  kPolicySkewed,
  kPolicySPT,
  kPolicyLPT
};

struct PolicySimOptions {
  pdlfs::Env* env;
  const char* prof_dir;
};

class PolicySim {
 public:
  explicit PolicySim(const PolicySimOptions& options)
      : options_(options), env_(options.env) {}

  void Run() {
    InitializePolicies();
    SimulateTrace();
    LogSummary();
  }

  void InitializePolicies();

  void SimulateTrace();

  void LogSummary() {
    logf(LOG_INFO, "\n\nFinished trace replay. Summary:\n");
    for (auto& ctx : policies_) ctx.LogSummary();
  }

  void InvokePolicies(std::vector<int>& cost_actual);

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
};
}  // namespace amr