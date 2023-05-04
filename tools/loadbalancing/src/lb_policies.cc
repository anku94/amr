//
// Created by Ankush J on 4/10/23.
//

#include "lb_policies.h"

#include "policy.h"

namespace amr {
std::string LoadBalancePolicies::PolicyToString(LoadBalancingPolicy policy) {
  switch (policy) {
    case LoadBalancingPolicy::kPolicyContiguous:
      return "Contiguous";
    case LoadBalancingPolicy::kPolicySkewed:
      return "Skewed";
    case LoadBalancingPolicy::kPolicyRoundRobin:
      return "RoundRobin";
    case LoadBalancingPolicy::kPolicySPT:
      return "SPT";
    case LoadBalancingPolicy::kPolicyLPT:
      return "LPT";
    case LoadBalancingPolicy::kPolicyILP:
      return "ILP";
  }

  logf(LOG_ERRO, "LoadBalancingPolicy not implemented!");
  ABORT("Unknown LoadBalancingPolicy");
  return "Unknown LoadBalancingPolicy";
}

int LoadBalancePolicies::AssignBlocksInternal(
    LoadBalancingPolicy policy, std::vector<double> const& costlist,
    std::vector<int>& ranklist, int nranks) {
  std::string policy_str = PolicyToString(policy);
  logf(LOG_DBG2, "[LoadBalancePolicies] Assignment LoadBalancingPolicy: %s",
       policy_str.c_str());

  ranklist.resize(costlist.size());

  switch (policy) {
    case LoadBalancingPolicy::kPolicyContiguous:
      return AssignBlocksContiguous(costlist, ranklist, nranks);
    case LoadBalancingPolicy::kPolicySkewed:
      return AssignBlocksSkewed(costlist, ranklist, nranks);
    case LoadBalancingPolicy::kPolicyRoundRobin:
      return AssignBlocksRoundRobin(costlist, ranklist, nranks);
    case LoadBalancingPolicy::kPolicySPT:
      return AssignBlocksSPT(costlist, ranklist, nranks);
    case LoadBalancingPolicy::kPolicyLPT:
      return AssignBlocksLPT(costlist, ranklist, nranks);
    case LoadBalancingPolicy::kPolicyILP:
      return AssignBlocksILP(costlist, ranklist, nranks);
    default:
      ABORT("LoadBalancingPolicy not implemented!!");
  }

  return -1;
}

int LoadBalancePolicies::AssignBlocksRoundRobin(
    const std::vector<double>& costlist, std::vector<int>& ranklist,
    int nranks) {
  for (int block_id = 0; block_id < costlist.size(); block_id++) {
    int block_rank = block_id % nranks;
    logf(LOG_DBG2, "Block %d: Rank %d", block_id, block_rank);

    ranklist[block_id] = block_rank;
  }

  return 0;
}

int LoadBalancePolicies::AssignBlocksSkewed(
    const std::vector<double>& costlist, std::vector<int>& ranklist,
    int nranks) {
  int nblocks = costlist.size();

  float avg_alloc   = nblocks * 1.0f / nranks;
  int   rank0_alloc = ceilf(avg_alloc);

  while ((nblocks - rank0_alloc) % (nranks - 1)) {
    rank0_alloc++;
  }

  if (rank0_alloc >= nblocks) {
    std::stringstream msg;
    msg << "### FATAL ERROR rank0_alloc >= nblocks "
        << "(" << rank0_alloc << ", " << nblocks << ")" << std::endl;
    ABORT(msg.str().c_str());
  }

  for (int bid = 0; bid < nblocks; bid++) {
    if (bid <= rank0_alloc) {
      ranklist[bid] = 0;
    } else {
      int rem_alloc = (nblocks - rank0_alloc) / (nranks - 1);
      int bid_adj   = bid - rank0_alloc;
      ranklist[bid] = 1 + bid_adj / rem_alloc;
    }
  }

  return 0;
}

int LoadBalancePolicies::AssignBlocksContiguous(
    const std::vector<double>& costlist, std::vector<int>& ranklist,
    int nranks) {
  double const total_cost =
      std::accumulate(costlist.begin(), costlist.end(), 0.0);

  int    rank           = nranks - 1;
  double target_cost    = total_cost / nranks;
  double my_cost        = 0.0;
  double remaining_cost = total_cost;
  // create rank list from the end: the master MPI rank should have less load
  for (int block_id = costlist.size() - 1; block_id >= 0; block_id--) {
    if (target_cost == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock" << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks."
          << std::endl;
      logf(LOG_WARN, "%s", msg.str().c_str());
//      ABORT(msg.str().c_str());
      logf(LOG_WARN, "Thugs don't abort on fatal errors.");
      return -1;
    }
    my_cost += costlist[block_id];
    ranklist[block_id] = rank;
    if (my_cost >= target_cost && rank > 0) {
      rank--;
      remaining_cost -= my_cost;
      my_cost     = 0.0;
      target_cost = remaining_cost / (rank + 1);
    }
  }

  return 0;
}
}  // namespace amr