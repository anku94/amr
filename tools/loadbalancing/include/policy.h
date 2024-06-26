//
// Created by Ankush J on 4/11/23.
//

#pragma once

#include <regex>
#include <string>
#include <vector>

#include "lb_policies.h"

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
  kPolicyContigImproved2,
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

struct LBPolicyWithOpts;

struct PolicyExecOpts {
  const char* policy_name;
  const char* policy_id;

  CostEstimationPolicy cost_policy;
  TriggerPolicy trigger_policy;

  const char* output_dir;
  pdlfs::Env* env;

  int nranks;
  int nblocks_init;

  int cache_ttl;
  int trigger_interval;

 public:
  PolicyExecOpts()
      : policy_name("<undefined>")
      , policy_id("<undefined>")
      , cost_policy(CostEstimationPolicy::kUnitCost)
      , trigger_policy(TriggerPolicy::kEveryTimestep)
      , output_dir(nullptr)
      , env(nullptr)
      , nranks(0)
      , nblocks_init(0)
      , cache_ttl(15)
      , trigger_interval(100) {}

  void SetPolicy(const char* name, const char* id, CostEstimationPolicy cep,
                 TriggerPolicy tp) {
    policy_name = name;
    policy_id = id;
    cost_policy = cep;
    trigger_policy = tp;
  }

  void SetPolicy(const char* name, const char* id, CostEstimationPolicy cep) {
    policy_name = name;
    policy_id = id;
    cost_policy = cep;
    trigger_policy = TriggerPolicy::kUnspecified;
  }
};
}  // namespace amr
