//
// Created by Ankush J on 5/4/23.
//

#include "policy.h"

namespace amr {
std::string PolicyToString(LoadBalancingPolicy policy) {
  switch (policy) {
    case LoadBalancingPolicy::kPolicyContiguous:
      return "Contiguous";
    case LoadBalancingPolicy::kPolicyRoundRobin:
      return "RoundRobin";
    case LoadBalancingPolicy::kPolicySkewed:
      return "Skewed";
    case LoadBalancingPolicy::kPolicySPT:
      return "SPT";
    case LoadBalancingPolicy::kPolicyLPT:
      return "LPT";
    case LoadBalancingPolicy::kPolicyILP:
      return "ILP";
    default:
      return "<undefined>";
  }
}

std::string PolicyToString(CostEstimationPolicy policy) {
  switch (policy) {
    case CostEstimationPolicy::kUnitCost:
      return "UnitCost";
    case CostEstimationPolicy::kExtrapolatedCost:
      return "ExtrapolatedCost";
    case CostEstimationPolicy::kCachedExtrapolatedCost:
      return "CachedExtrapolatedCost";
    case CostEstimationPolicy::kOracleCost:
      return "OracleCost";
    default:
      return "<undefined>";
  }
}

std::string PolicyToString(TriggerPolicy policy) {
  switch (policy) {
    case TriggerPolicy::kEveryTimestep:
      return "EveryTimestep";
    case TriggerPolicy::kOnMeshChange:
      return "OnMeshChange";
    default:
      return "<undefined>";
  }
}
}  // namespace amr
