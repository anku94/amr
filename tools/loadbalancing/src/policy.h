//
// Created by Ankush J on 4/11/23.
//

#pragma once

#include <pdlfs-common/env.h>
namespace amr {
enum class LoadBalancingPolicy {
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
};

enum class TriggerPolicy { kEveryTimestep, kOnMeshChange };

std::string PolicyToString(LoadBalancingPolicy policy);

std::string PolicyToString(CostEstimationPolicy policy);

std::string PolicyToString(TriggerPolicy policy);

struct PolicyExecOpts {
  const char* policy_name;

  LoadBalancingPolicy lb_policy;
  CostEstimationPolicy cost_policy;
  TriggerPolicy trigger_policy;

  const char* output_dir;
  pdlfs::Env* env;

  int nranks;
  int nblocks_init;

  PolicyExecOpts()
      : policy_name("<undefined>"),
        lb_policy(LoadBalancingPolicy::kPolicyContiguous),
        cost_policy(CostEstimationPolicy::kUnitCost),
        trigger_policy(TriggerPolicy::kEveryTimestep),
        output_dir(nullptr),
        env(nullptr),
        nranks(0),
        nblocks_init(0) {}

  void SetPolicy(const char* name, LoadBalancingPolicy lp,
                 CostEstimationPolicy cep, TriggerPolicy tp) {
    policy_name = name;
    lb_policy = lp;
    cost_policy = cep;
    trigger_policy = tp;
  }
};
}  // namespace amr