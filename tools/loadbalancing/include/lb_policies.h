//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <mpi.h>

#include <vector>

namespace amr {

struct PolicyOptsCDPI;
struct PolicyOptsHybridCDPFirst;
struct PolicyOptsHybrid;
struct PolicyOptsILP;
struct PolicyOptsChunked;

enum class LoadBalancePolicy;

class LoadBalancePolicies {
 public:
  static int AssignBlocks(const char* policy_name,
                          std::vector<double> const& costlist,
                          std::vector<int>& ranklist, int nranks);

  static int AssignBlocksParallel(const char* policy_name,
                                  std::vector<double> const& costlist,
                                  std::vector<int>& ranklist, MPI_Comm comm);

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
                                            MPI_Comm comm, int my_rank,
                                            int nranks,
                                            PolicyOptsChunked const& opts);

  friend class LoadBalancingPoliciesTest;
  friend class PolicyTest;
  friend class LBChunkwise;
  //  friend class PolicyExecCtx;
  //  friend class ScaleExecCtx;
};

class LBChunkwise;
}  // namespace amr
