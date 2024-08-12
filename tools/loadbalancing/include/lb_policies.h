//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <mpi.h>

#include <vector>

#include "policy_wopts.h"

namespace amr {

struct PolicyOptsCDPI;
struct PolicyOptsHybridCDPFirst;
struct PolicyOptsHybrid;
struct PolicyOptsILP;
struct PolicyOptsChunked;

enum class LoadBalancePolicy;

class LoadBalancePolicies {
 public:
  //
  // Top-level function over AssignBlocks and AssignBlocksParallel.
  // Both AssignBlocks and AssignBlocksParallel follow their own rules for
  // calling each other for hybrid policies, but
  // AssignBlocksCached will only call one of those functions.
  //
  // There exists no callpath in which AssignBlocksCached ends up calling
  // itself.
  //
  // If a policy is configured with the parameter skip_cache,
  // the cache will be bypassed.
  //
  static int AssignBlocksCached(const char* policy_name,
                                std::vector<double> const& costlist,
                                std::vector<int>& ranklist, int nranks,
                                int my_rank = 0, MPI_Comm comm = MPI_COMM_NULL);

  static int AssignBlocks(const LBPolicyWithOpts& policy,
                          std::vector<double> const& costlist,
                          std::vector<int>& ranklist, int nranks);

  //
  // AssignBlocksParallel: Use multiple MPI ranks to compute assignment
  // This will only use the parallel implementation for certain
  // policies, currently cdpc512, and defer to AssignBlocks for the rest.
  //
  // It does not use nranks for assignment but not for parallelism.
  // For parallelism, it uses the number of ranks in the communicator.
  // This allows for different placement sizes to be tested using smaller
  // communicator sizes.
  //
  static int AssignBlocksParallel(const LBPolicyWithOpts& policy,
                                  std::vector<double> const& costlist,
                                  std::vector<int>& ranklist, int nranks,
                                  MPI_Comm comm);

 private:
  static int AssignBlocksRoundRobin(std::vector<double> const& costlist,
                                    std::vector<int>& ranklist, int nranks);

  static int AssignBlocksSkewed(std::vector<double> const& costlist,
                                std::vector<int>& ranklist, int nranks);

  static int AssignBlocksContiguous(std::vector<double> const& costlist,
                                    std::vector<int>& ranklist, int nranks);

  static int AssignBlocksSPT(std::vector<double> const& costlist,
                             std::vector<int>& ranklist, int nranks);

  static int AssignBlocksLPT(std::vector<double> const& costlist,
                             std::vector<int>& ranklist, int nranks);

  static int AssignBlocksILP(std::vector<double> const& costlist,
                             std::vector<int>& ranklist, int nranks,
                             PolicyOptsILP const& opts);

  static int AssignBlocksContigImproved(std::vector<double> const& costlist,
                                        std::vector<int>& ranklist, int nranks);

  static int AssignBlocksContigImproved2(std::vector<double> const& costlist,
                                         std::vector<int>& ranklist,
                                         int nranks);

  static int AssignBlocksCppIter(std::vector<double> const& costlist,
                                 std::vector<int>& ranklist, int nranks,
                                 PolicyOptsCDPI const& opts);

  static int AssignBlocksHybrid(std::vector<double> const& costlist,
                                std::vector<int>& ranklist, int nranks,
                                PolicyOptsHybrid const& opts);

  static int AssignBlocksHybridCppFirst(std::vector<double> const& costlist,
                                        std::vector<int>& ranklist, int nranks,
                                        PolicyOptsHybridCDPFirst const& opts);

  static int AssignBlocksCDPChunked(std::vector<double> const& costlist,
                                    std::vector<int>& ranklist, int nranks,
                                    PolicyOptsChunked const& opts);

  static int AssignBlocksParallelCDPChunked(std::vector<double> const& costlist,
                                            std::vector<int>& ranklist,
                                            int nranks,
                                            PolicyOptsChunked const& opts,
                                            MPI_Comm comm, int mympirank,
                                            int nmpiranks);

  static int AssignBlocksParallelHybridCDPFirst(
      std::vector<double> const& costlist, std::vector<int>& ranklist,
      int nranks, PolicyOptsHybridCDPFirst const& opts, MPI_Comm comm,
      int mympirank, int nmpiranks);

  friend class LoadBalancingPoliciesTest;
  friend class PolicyTest;
  friend class LBChunkwise;
  //  friend class PolicyExecCtx;
  //  friend class ScaleExecCtx;
};

class LBChunkwise;
}  // namespace amr
