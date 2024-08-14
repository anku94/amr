//
// Created by Ankush J on 4/10/23.
//

#include "lb_policies.h"

#include "assignment_cache.h"
#include "common.h"
#include "constants.h"
#include "policy.h"
#include "policy_utils.h"
#include "policy_wopts.h"

namespace amr {
int LoadBalancePolicies::AssignBlocksCached(const char* policy_name,
                                            std::vector<double> const& costlist,
                                            std::vector<int>& ranklist,
                                            int nranks, int my_rank,
                                            MPI_Comm comm) {
  if (my_rank != 0) {
    decrement_log_level_once();
  }

  static AssignmentCache cache(Constants::kMaxAssignmentCacheReuse);
  int rv = 0;

  auto& policy = PolicyUtils::GetPolicy(policy_name);

  if (policy.skip_cache) {
    if (my_rank == 0) {
      logv(__LOG_ARGS__, LOG_INFO, "Skipping cache");
    }

    bool cache_ret = cache.Get(costlist.size(), ranklist);
    if (cache_ret) {
      logv(__LOG_ARGS__, LOG_DBUG, "Cache hit");
      return 0;
    }
  }

  if (comm == MPI_COMM_NULL) {
    rv = AssignBlocks(policy, costlist, ranklist, nranks);
  } else {
    rv = AssignBlocksParallel(policy, costlist, ranklist, nranks, comm);
  }

  PolicyUtils::LogAssignmentStats(costlist, ranklist, nranks, my_rank);

  cache.Put(costlist.size(), ranklist);

  return rv;
}

int LoadBalancePolicies::AssignBlocks(const LBPolicyWithOpts& policy,
                                      std::vector<double> const& costlist,
                                      std::vector<int>& ranklist, int nranks) {
  ranklist.resize(costlist.size());

  // const LBPolicyWithOpts& policy = PolicyUtils::GetPolicy(policy_name);

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
    const LBPolicyWithOpts& policy, std::vector<double> const& costlist,
    std::vector<int>& ranklist, int nranks, MPI_Comm comm) {
  static int mympirank = -1;
  static int nmpiranks = -1;

  if (mympirank == -1 or nmpiranks == -1) {
    MPI_Comm_rank(comm, &mympirank);
    MPI_Comm_size(comm, &nmpiranks);
  }

  ranklist.resize(costlist.size());
  // const LBPolicyWithOpts& policy = PolicyUtils::GetPolicy(policy_name);

  switch (policy.policy) {
    case LoadBalancePolicy::kPolicyCDPChunked:
      return AssignBlocksParallelCDPChunked(costlist, ranklist, nranks,
                                            policy.chunked_opts, comm,
                                            mympirank, nmpiranks);
    case LoadBalancePolicy::kPolicyHybridCppFirstV2:
      return AssignBlocksParallelHybridCDPFirst(costlist, ranklist, nranks,
                                                policy.hcf_opts, comm,
                                                mympirank, nmpiranks);
    default:
      return AssignBlocks(policy, costlist, ranklist, nranks);
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
  int nblocks = costlist.size();
  if (nblocks < nranks) {
    std::stringstream msg;
    msg << "### FATAL ERROR in AssignBlocksContiguous" << std::endl
        << "nblocks < nranks" << "(" << nblocks << ", " << nranks << ")"
        << std::endl;
    ABORT(msg.str().c_str());
  }

  double const total_cost =
      std::accumulate(costlist.begin(), costlist.end(), 0.0);

  int rank = nranks - 1;
  double target_cost = total_cost / nranks;
  double my_cost = 0.0;
  double remaining_cost = total_cost;
  // create rank list from the end: the master MPI rank should have less load
  for (int block_id = nblocks - 1; block_id >= 0; block_id--) {
    if (target_cost == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock" << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks."
          << std::endl;
      logv(__LOG_ARGS__, LOG_WARN, "%s", msg.str().c_str());
      ABORT(msg.str().c_str());
      // logv(__LOG_ARGS__, LOG_WARN, "Thugs don't abort on fatal errors.");
      return -1;
    }
    my_cost += costlist[block_id];
    ranklist[block_id] = rank;

    // all conditions
    bool filled_enough = (my_cost >= target_cost);
    // remaining blocks == remaining ranks
    bool low_on_blocks = (block_id == rank);
    if ((filled_enough or low_on_blocks) && (rank > 0)) {
      rank--;
      remaining_cost -= my_cost;
      my_cost = 0.0;
      target_cost = remaining_cost / (rank + 1);
    }
  }

  return 0;
}
}  // namespace amr
