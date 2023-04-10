#pragma once

#include "lb_policies.h"
#include "prof_set_reader.h"

#include "pdlfs-common/env.h"

#include <regex>
#include <vector>

#define PROF_DIR "/mnt/ltio/parthenon-topo/profile20/"

namespace amr {
enum class Policy { kPolicyContiguous, kPolicyRoundRobin, kPolicySkewed, kPolicySPT, kPolicyLPT };

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

  double excess_cost_;
  double total_cost_avg_;
  double total_cost_max_;
};
}  // namespace amr
