//
// Created by Ankush J on 4/11/23.
//

#pragma once

#include "lb_policies.h"

#include <regex>
#include <string>
#include <vector>

namespace pdlfs {
class Env;
}

namespace amr {
enum class ProfTimeCombinePolicy { kUseFirst, kUseLast, kAdd };

enum class LoadBalancePolicy {
  kPolicyActual,
  kPolicyContiguousUnitCost,
  kPolicyContiguousActualCost,
  kPolicyRoundRobin,
  kPolicySkewed,
  kPolicySPT,
  kPolicyLPT,
  kPolicyILP,
  kPolicyContigImproved,
  kPolicyCppIter,
  kPolicyHybrid,
  kPolicyHybridCppFirst,
  kPolicyHybridCppFirstV2
};

/** Policy kUnitCost is not really necessary
 * as the LB policy kPolicyContiguousUnitCost implicitly includes it
 * Keeping this here anyway.
 */
enum class CostEstimationPolicy {
  kUnspecified,
  kUnitCost,
  kExtrapolatedCost,
  kOracleCost,
  kCachedExtrapolatedCost,
};

enum class TriggerPolicy {
  kUnspecified,
  kEveryTimestep,
  kEveryNTimesteps,
  kOnMeshChange
};

class PolicyUtils {
 public:
  static LoadBalancePolicy StringToPolicy(std::string const& policy_str);

  static std::string PolicyToString(LoadBalancePolicy policy);

  static std::string PolicyToString(CostEstimationPolicy policy);

  static std::string PolicyToString(TriggerPolicy policy);

  static void ExtrapolateCosts(std::vector<double> const& costs_prev,
                               std::vector<int>& refs, std::vector<int>& derefs,
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
};

struct PolicyExecOpts {
  const char* policy_name;

  LoadBalancePolicy lb_policy;
  CostEstimationPolicy cost_policy;
  TriggerPolicy trigger_policy;

  const char* output_dir;
  pdlfs::Env* env;

  int nranks;
  int nblocks_init;

  int cache_ttl;
  int trigger_interval;

 private:
  PolicyOptsILP lb_opts_ilp;

 public:
  PolicyExecOpts()
      : policy_name("<undefined>"),
        lb_policy(LoadBalancePolicy::kPolicyContiguousActualCost),
        cost_policy(CostEstimationPolicy::kUnitCost),
        trigger_policy(TriggerPolicy::kEveryTimestep),
        output_dir(nullptr),
        env(nullptr),
        nranks(0),
        nblocks_init(0),
        cache_ttl(15),
        trigger_interval(100) {}

  void SetPolicy(const char* name, LoadBalancePolicy lp,
                 CostEstimationPolicy cep, TriggerPolicy tp) {
    policy_name = name;
    lb_policy = lp;
    cost_policy = cep;
    trigger_policy = tp;
  }

  void SetPolicy(const char* name, LoadBalancePolicy lp,
                 CostEstimationPolicy cep) {
    policy_name = name;
    lb_policy = lp;
    cost_policy = cep;
    trigger_policy = TriggerPolicy::kUnspecified;
  }

  void SetLBOpts(PolicyOptsILP& opts) { lb_opts_ilp = opts; }

  void* GetLBOpts() const { return (void*)&lb_opts_ilp; }
};
}  // namespace amr
