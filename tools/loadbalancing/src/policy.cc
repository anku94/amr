//
// Created by Ankush J on 5/4/23.
//

#include "policy.h"

#include <algorithm>
#include <cassert>
#include <string>

namespace amr {
std::string PolicyUtils::PolicyToString(LoadBalancePolicy policy) {
  switch (policy) {
    case LoadBalancePolicy::kPolicyContiguousActualCost:
      return "Contiguous";
    case LoadBalancePolicy::kPolicyRoundRobin:
      return "RoundRobin";
    case LoadBalancePolicy::kPolicySkewed:
      return "Skewed";
    case LoadBalancePolicy::kPolicySPT:
      return "SPT";
    case LoadBalancePolicy::kPolicyLPT:
      return "LPT";
    case LoadBalancePolicy::kPolicyILP:
      return "ILP";
    default:
      return "<undefined>";
  }
}

std::string PolicyUtils::PolicyToString(CostEstimationPolicy policy) {
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

std::string PolicyUtils::PolicyToString(TriggerPolicy policy) {
  switch (policy) {
    case TriggerPolicy::kEveryTimestep:
      return "EveryTimestep";
    case TriggerPolicy::kOnMeshChange:
      return "OnMeshChange";
    default:
      return "<undefined>";
  }
}

void PolicyUtils::ExtrapolateCosts(std::vector<double> const& costs_prev,
                                   std::vector<int>& refs,
                                   std::vector<int>& derefs,
                                   std::vector<double>& costs_cur) {
  int nblocks_prev = costs_prev.size();
  int nblocks_cur = nblocks_prev + (refs.size() * 7) - (derefs.size() * 7 / 8);

  costs_cur.resize(0);
  std::sort(refs.begin(), refs.end());
  std::sort(derefs.begin(), derefs.end());

  int ref_idx = 0;
  int deref_idx = 0;
  for (int bidx = 0; bidx < nblocks_prev;) {
    if (ref_idx < refs.size() && refs[ref_idx] == bidx) {
      for (int dim = 0; dim < 8; dim++) {
        costs_cur.push_back(costs_prev[bidx]);
      }
      ref_idx++;
      bidx++;
    } else if (deref_idx < derefs.size() && derefs[deref_idx] == bidx) {
      double cost_deref_avg = 0;
      for (int dim = 0; dim < 8; dim++) {
        cost_deref_avg += costs_prev[bidx + dim];
      }
      cost_deref_avg /= 8;
      costs_cur.push_back(cost_deref_avg);
      deref_idx += 8;
      bidx += 8;
    } else {
      costs_cur.push_back(costs_prev[bidx]);
      bidx++;
    }
  }

  assert(costs_cur.size() == nblocks_cur);
}
}  // namespace amr
