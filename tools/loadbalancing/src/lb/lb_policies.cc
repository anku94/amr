//
// Created by Ankush J on 4/10/23.
//

#include "lb_policies.h"

#include "common.h"
#include "policy.h"
#include "policy_utils.h"
#include "policy_wopts.h"

namespace amr {
int LoadBalancePolicies::AssignBlocks(const char* policy_name,
                                      std::vector<double> const& costlist,
                                      std::vector<int>& ranklist, int nranks) {
  ranklist.resize(costlist.size());

  const LBPolicyWithOpts& policy = PolicyUtils::GetPolicy(policy_name);

  switch (policy.policy) {
    case LoadBalancePolicy::kPolicyActual:
      return 0;
    case LoadBalancePolicy::kPolicyContiguousUnitCost:
      return AssignBlocksContiguous(std::vector<double>(costlist.size(), 1.0),
                                    ranklist, nranks);
    case LoadBalancePolicy::kPolicyContiguousActualCost:
      return AssignBlocksContiguous(costlist, ranklist, nranks);
    case LoadBalancePolicy::kPolicySkewed:
      throw std::runtime_error("Skewed policy is deprecated");
    case LoadBalancePolicy::kPolicyRoundRobin:
      throw std::runtime_error("RoundRobin policy is deprecated");
    case LoadBalancePolicy::kPolicySPT:
      throw std::runtime_error("SPT policy is deprecated");
    case LoadBalancePolicy::kPolicyLPT:
      return AssignBlocksLPT(costlist, ranklist, nranks);
    case LoadBalancePolicy::kPolicyILP:
      return AssignBlocksILP(costlist, ranklist, nranks, policy.ilp_opts);
    case LoadBalancePolicy::kPolicyContigImproved:
      return AssignBlocksContigImproved(costlist, ranklist, nranks);
    case LoadBalancePolicy::kPolicyContigImproved2:
      throw std::runtime_error("ContigImproved2 policy is deprecated");
      return AssignBlocksContigImproved2(costlist, ranklist, nranks);
    case LoadBalancePolicy::kPolicyCppIter:
      return AssignBlocksCppIter(costlist, ranklist, nranks, policy.cdp_opts);
    case LoadBalancePolicy::kPolicyHybrid:
      throw std::runtime_error("Hybrid policy is deprecated");
    case LoadBalancePolicy::kPolicyHybridCppFirst:
      throw std::runtime_error("HybridCppFirst policy is deprecated");
    case LoadBalancePolicy::kPolicyHybridCppFirstV2:
      return AssignBlocksHybridCppFirst(costlist, ranklist, nranks,
                                        policy.hcf_opts);
    case LoadBalancePolicy::kPolicyCDPChunked:
      return AssignBlocksCDPChunked(costlist, ranklist, nranks,
                                    policy.chunked_opts);
    default:
      ABORT("LoadBalancePolicy not implemented!!");
  }
  return -1;
}

int LoadBalancePolicies::AssignBlocksParallel(
    const char* policy_name, std::vector<double> const& costlist,
    std::vector<int>& ranklist, MPI_Comm comm) {
  ranklist.resize(costlist.size());

  static int my_rank = -1;
  static int nranks = -1;

  if (my_rank == -1) {
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &nranks);
  }

  const LBPolicyWithOpts& policy = PolicyUtils::GetPolicy(policy_name);

  switch (policy.policy) {
    case LoadBalancePolicy::kPolicyCDPChunked:
      return AssignBlocksParallelCDPChunked(costlist, ranklist, comm, my_rank,
                                            nranks, policy.chunked_opts);
    default:
      return AssignBlocks(policy_name, costlist, ranklist, nranks);
  }
}

int LoadBalancePolicies::AssignBlocksRoundRobin(
    const std::vector<double>& costlist, std::vector<int>& ranklist,
    int nranks) {
  for (int block_id = 0; block_id < costlist.size(); block_id++) {
    int block_rank = block_id % nranks;
    ranklist[block_id] = block_rank;
  }

  return 0;
}

int LoadBalancePolicies::AssignBlocksSkewed(const std::vector<double>& costlist,
                                            std::vector<int>& ranklist,
                                            int nranks) {
  int nblocks = costlist.size();

  float avg_alloc = nblocks * 1.0f / nranks;
  int rank0_alloc = ceilf(avg_alloc);

  while ((nblocks - rank0_alloc) % (nranks - 1)) {
    rank0_alloc++;
  }

  if (rank0_alloc >= nblocks) {
    std::stringstream msg;
    msg << "### FATAL ERROR rank0_alloc >= nblocks " << "(" << rank0_alloc
        << ", " << nblocks << ")" << std::endl;
    ABORT(msg.str().c_str());
  }

  for (int bid = 0; bid < nblocks; bid++) {
    if (bid <= rank0_alloc) {
      ranklist[bid] = 0;
    } else {
      int rem_alloc = (nblocks - rank0_alloc) / (nranks - 1);
      int bid_adj = bid - rank0_alloc;
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

  int rank = nranks - 1;
  double target_cost = total_cost / nranks;
  double my_cost = 0.0;
  double remaining_cost = total_cost;
  // create rank list from the end: the master MPI rank should have less load
  for (int block_id = costlist.size() - 1; block_id >= 0; block_id--) {
    if (target_cost == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock" << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks."
          << std::endl;
      logv(__LOG_ARGS__, LOG_WARN, "%s", msg.str().c_str());
      //      ABORT(msg.str().c_str());
      logv(__LOG_ARGS__, LOG_WARN, "Thugs don't abort on fatal errors.");
      return -1;
    }
    my_cost += costlist[block_id];
    ranklist[block_id] = rank;
    if (my_cost >= target_cost && rank > 0) {
      rank--;
      remaining_cost -= my_cost;
      my_cost = 0.0;
      target_cost = remaining_cost / (rank + 1);
    }
  }

  return 0;
}
}  // namespace amr
