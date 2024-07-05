#pragma once

#include <vector>

#include "lb_policies.h"
#include "policy.h"
#include "policy_wopts.h"

namespace amr {
// fwd decl
class MiscTest;

class PolicyUtils {
 public:
  static const LBPolicyWithOpts GetPolicy(const char* policy_name);

  static std::string PolicyToString(LoadBalancePolicy policy);

  static std::string PolicyToString(CostEstimationPolicy policy);

  static std::string PolicyToString(TriggerPolicy policy);

  static void ExtrapolateCosts2D(std::vector<double> const& costs_prev,
                                 std::vector<int>& refs,
                                 std::vector<int>& derefs,
                                 std::vector<double>& costs_cur);

  static void ExtrapolateCosts3D(std::vector<double> const& costs_prev,
                                 std::vector<int>& refs,
                                 std::vector<int>& derefs,
                                 std::vector<double>& costs_cur);

  static void ComputePolicyCosts(int nranks,
                                 std::vector<double> const& cost_list,
                                 std::vector<int> const& rank_list,
                                 std::vector<double>& rank_times,
                                 double& rank_time_avg, double& rank_time_max);

  static double ComputeLocCost(std::vector<int> const& rank_list);

  static std::string GetSafePolicyName(const char* policy_name) {
    std::string result = policy_name;

    std::regex rm_unsafe("[/-]");
    result = std::regex_replace(result, rm_unsafe, "_");

    std::regex clean_suffix("actual_cost");
    result = std::regex_replace(result, clean_suffix, "ac");

    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
  }

  static std::string GetLogPath(const char* output_dir, const char* policy_name,
                                const char* suffix);

 private:
  static LoadBalancePolicy StringToPolicy(std::string const& policy_str);

  static const LBPolicyWithOpts GenHybrid(const std::string& policy_str);

  static const std::map<std::string, LBPolicyWithOpts> kPolicyMap;

  friend class MiscTest;
};
}  // namespace amr
