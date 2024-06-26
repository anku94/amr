//
// Created by Ankush J on 5/4/23.
//

#include "policy_utils.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>

#include "common.h"
#include "constants.h"
#include "policy.h"
#include "policy_wopts.h"

namespace amr {
const std::map<std::string, LBPolicyWithOpts> PolicyUtils::kPolicyMap = {
    {"actual",
     {.id = "actual",
      .name = "Actual",
      .policy = LoadBalancePolicy::kPolicyActual}},
    {"baseline",
     {.id = "baseline",
      .name = "Baseline",
      .policy = LoadBalancePolicy::kPolicyContiguousUnitCost}},
    {"lpt",
     {.id = "lpt", .name = "LPT", .policy = LoadBalancePolicy::kPolicyLPT}},
    {"cdp",
     {.id = "cdp",
      .name = "Contiguous-DP (CDP)",
      .policy = LoadBalancePolicy::kPolicyContigImproved}},
    {"cdpi50",
     {.id = "cdpi50",
      .name = "CDP-I50",
      .policy = LoadBalancePolicy::kPolicyCppIter,
      .cdp_opts = {.niter_frac = 0, .niters = 50}}},
    {"cdpi250",
     {.id = "cdpi250",
      .name = "CDP-I250",
      .policy = LoadBalancePolicy::kPolicyCppIter,
      .cdp_opts = {.niter_frac = 0, .niters = 250}}},
};

const LBPolicyWithOpts PolicyUtils::GetPolicy(const char* policy_name) {
  std::string policy_str(policy_name);

  if (policy_str.substr(0, 6) == "hybrid") {
    return GenHybrid(policy_str);
  }

  if (kPolicyMap.find(policy_name) == kPolicyMap.end()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetPolicy" << std::endl
        << "Policy " << policy_name << " not found in kPolicyMap" << std::endl;
    ABORT(msg.str().c_str());
  }

  return kPolicyMap.at(policy_name);
}
LoadBalancePolicy PolicyUtils::StringToPolicy(std::string const& policy_str) {
  if (policy_str == "baseline") {
    return LoadBalancePolicy::kPolicyContiguousUnitCost;
  } else if (policy_str == "lpt") {
    return LoadBalancePolicy::kPolicyLPT;
  } else if (policy_str == "ci") {
    return LoadBalancePolicy::kPolicyContigImproved;
  } else if (policy_str == "cdpp") {
    return LoadBalancePolicy::kPolicyCppIter;
  } else if (policy_str == "hybrid") {
    return LoadBalancePolicy::kPolicyHybridCppFirstV2;
  } else if (policy_str == "hybrid02") {
    return LoadBalancePolicy::kPolicyHybridCppFirstV2;
  }

  throw std::runtime_error("Unknown policy string: " + policy_str);

  return LoadBalancePolicy::kPolicyContiguousUnitCost;
}

std::string PolicyUtils::PolicyToString(LoadBalancePolicy policy) {
  switch (policy) {
    case LoadBalancePolicy::kPolicyContiguousUnitCost:
      return "Baseline";
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
    case LoadBalancePolicy::kPolicyContigImproved:
      return "kContigImproved";
    case LoadBalancePolicy::kPolicyCppIter:
      return "kContig++Iter";
    case LoadBalancePolicy::kPolicyILP:
      return "ILP";
    case LoadBalancePolicy::kPolicyHybrid:
      return "Hybrid";
    case LoadBalancePolicy::kPolicyHybridCppFirst:
      return "HybridCppFirst";
    case LoadBalancePolicy::kPolicyHybridCppFirstV2:
      return "HybridCppFirstV2";
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
    case TriggerPolicy::kEveryNTimesteps:
      return "EveryNTimesteps";
    case TriggerPolicy::kEveryTimestep:
      return "EveryTimestep";
    case TriggerPolicy::kOnMeshChange:
      return "OnMeshChange";
    default:
      return "<undefined>";
  }
}

void PolicyUtils::ExtrapolateCosts2D(std::vector<double> const& costs_prev,
                                     std::vector<int>& refs,
                                     std::vector<int>& derefs,
                                     std::vector<double>& costs_cur) {
  static bool first_time = true;
  if (first_time) {
    logv(__LOG_ARGS__, LOG_WARN, "ALERT: Using 2D extrapolation!");
    first_time = false;
  }

  int nblocks_prev = costs_prev.size();
  int nblocks_cur = nblocks_prev + (refs.size() * 3) - (derefs.size() * 3 / 4);

  costs_cur.resize(0);
  std::sort(refs.begin(), refs.end());
  std::sort(derefs.begin(), derefs.end());

  int ref_idx = 0;
  int deref_idx = 0;
  for (int bidx = 0; bidx < nblocks_prev;) {
    if (ref_idx < refs.size() && refs[ref_idx] == bidx) {
      for (int dim = 0; dim < 4; dim++) {
        costs_cur.push_back(costs_prev[bidx]);
      }
      ref_idx++;
      bidx++;
    } else if (deref_idx < derefs.size() && derefs[deref_idx] == bidx) {
      double cost_deref_avg = 0;
      for (int dim = 0; dim < 4; dim++) {
        cost_deref_avg += costs_prev[bidx + dim];
      }
      cost_deref_avg /= 4;
      costs_cur.push_back(cost_deref_avg);
      deref_idx += 4;
      bidx += 4;
    } else {
      costs_cur.push_back(costs_prev[bidx]);
      bidx++;
    }
  }

  assert(costs_cur.size() == nblocks_cur);
}

void PolicyUtils::ExtrapolateCosts3D(std::vector<double> const& costs_prev,
                                     std::vector<int>& refs,
                                     std::vector<int>& derefs,
                                     std::vector<double>& costs_cur) {
  static bool first_time = true;
  if (first_time) {
    logv(__LOG_ARGS__, LOG_WARN, "ALERT: Using 3D extrapolation!");
    first_time = false;
  }

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

void PolicyUtils::ComputePolicyCosts(int nranks,
                                     std::vector<double> const& cost_list,
                                     std::vector<int> const& rank_list,
                                     std::vector<double>& rank_times,
                                     double& rank_time_avg,
                                     double& rank_time_max) {
  rank_times.clear();
  rank_times.resize(nranks, 0);
  int nblocks = cost_list.size();

  for (int bid = 0; bid < nblocks; bid++) {
    int block_rank = rank_list[bid];
    rank_times[block_rank] += cost_list[bid];
  }

  int const& (*max_func)(int const&, int const&) = std::max<int>;
  rank_time_max = std::accumulate(rank_times.begin(), rank_times.end(),
                                  rank_times.front(), max_func);
  uint64_t rtsum = std::accumulate(rank_times.begin(), rank_times.end(), 0ull);
  rank_time_avg = rtsum * 1.0 / nranks;
}

//
// Use an arbitary model to compute
// Intuition: amount of linear locality captured (lower is better)
// cost of 1 for neighboring ranks
// cost of 2 for same node (hardcoded rn)
// cost of 3 for arbitrary communication
//
double PolicyUtils::ComputeLocCost(std::vector<int> const& rank_list) {
  int nb = rank_list.size();
  int local_score = 0;

  for (int bidx = 0; bidx < nb - 1; bidx++) {
    int p = rank_list[bidx];
    int q = rank_list[bidx + 1];

    // Nodes for p and q, computed using assumptions
    int pn = p / Constants::kRanksPerNode;
    int qn = q / Constants::kRanksPerNode;

    if (p == q) {
      // nothing
    } else if (abs(q - p) == 1) {
      local_score += 1;
    } else if (qn == pn) {
      local_score += 2;
    } else {
      local_score += 3;
    }
  }

  double norm_score = local_score * 1.0 / nb;
  return norm_score;
}

std::string PolicyUtils::GetLogPath(const char* output_dir,
                                    const char* policy_name,
                                    const char* suffix) {
  std::string result = GetSafePolicyName(policy_name);
  result = std::string(output_dir) + "/" + result + "." + suffix;
  logv(__LOG_ARGS__, LOG_DBUG, "LoadBalancePolicy Name: %s, Log Fname: %s",
       policy_name, result.c_str());
  return result;
}

const LBPolicyWithOpts PolicyUtils::GenHybrid(const std::string& policy_str) {
  // policy_name = hybridX, where X is to be parsed into lpt_frac
  // parse policy_name, throw error if not in the correct format

  if (policy_str.substr(0, 6) != "hybrid") {
    std::stringstream msg;
    msg << "### FATAL ERROR in GenHybrid" << std::endl
        << "Policy " << policy_str << " not in the correct format" << std::endl;
    ABORT(msg.str().c_str());
  }

  int lpt_frac = std::stoi(policy_str.substr(6));
  if (lpt_frac < 0 or lpt_frac > 100) {
    std::stringstream msg;
    msg << "### FATAL ERROR in GenHybrid" << std::endl
        << "Policy " << policy_str << " not in the correct format" << std::endl;
    ABORT(msg.str().c_str());
  }

  std::string policy_name_friendly =
      "Hybrid (" + std::to_string(lpt_frac) + "%)";

  LBPolicyWithOpts policy = {
      .id = policy_str,
      .name = policy_name_friendly,
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = lpt_frac / 100.0}};

  logv(__LOG_ARGS__, LOG_DBUG, "Generated policy: %s",
       policy_name_friendly.c_str());

  return policy;
}

}  // namespace amr
