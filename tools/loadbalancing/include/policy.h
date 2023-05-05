//
// Created by Ankush J on 4/11/23.
//

#pragma once

#include <string>
#include <vector>

namespace pdlfs {
class Env;
}

namespace amr {
enum class LoadBalancePolicy {
  kPolicyContiguous,
  kPolicyRoundRobin,
  kPolicySkewed,
  kPolicySPT,
  kPolicyLPT,
  kPolicyILP
};

enum class CostEstimationPolicy {
  kUnitCost,
  kExtrapolatedCost,
  kOracleCost,
  kCachedExtrapolatedCost
};

enum class TriggerPolicy { kEveryTimestep, kOnMeshChange };

class PolicyUtils {
 public:
  static std::string PolicyToString(LoadBalancePolicy policy);

  static std::string PolicyToString(CostEstimationPolicy policy);

  static std::string PolicyToString(TriggerPolicy policy);

  static void ExtrapolateCosts(std::vector<double> const& costs_prev,
                               std::vector<int>& refs, std::vector<int>& derefs,
                               std::vector<double>& costs_cur);
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

  PolicyExecOpts()
      : policy_name("<undefined>"),
        lb_policy(LoadBalancePolicy::kPolicyContiguous),
        cost_policy(CostEstimationPolicy::kUnitCost),
        trigger_policy(TriggerPolicy::kEveryTimestep),
        output_dir(nullptr),
        env(nullptr),
        nranks(0),
        nblocks_init(0) {}

  void SetPolicy(const char* name, LoadBalancePolicy lp,
                 CostEstimationPolicy cep, TriggerPolicy tp) {
    policy_name = name;
    lb_policy = lp;
    cost_policy = cep;
    trigger_policy = tp;
  }
};
}  // namespace amr
