#pragma once

#include "lb_policies.h"
#include "prof_set_reader.h"

#include "pdlfs-common/env.h"

#include <regex>
#include <vector>

#define PROF_DIR "/mnt/ltio/parthenon-topo/profile20/"

namespace amr {
enum class Policy { kPolicyContiguous, kPolicyRoundRobin, kPolicySkewed };

struct PolicySimOptions {
  pdlfs::Env* env;
  const char* prof_dir;
  Policy policy;
};

class PolicySim {
 public:
  PolicySim(const PolicySimOptions& options)
      : options_(options),
        env_(options.env),
        excess_cost_(0),
        total_cost_avg_(0),
        total_cost_max_(0) {}

  void Run() { SimulateTrace(); }

  void SimulateTrace();

  void CheckAssignment(std::vector<int>& times);

 private:
  std::vector<std::string> LocateRelevantFiles(const std::string& root_dir) {
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

  double excess_cost_;
  double total_cost_avg_;
  double total_cost_max_;
};
}  // namespace amr
