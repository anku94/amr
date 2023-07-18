//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace amr {
enum class LoadBalancePolicy;
class LoadBalancePolicies {
 public:
  static int AssignBlocks(LoadBalancePolicy policy,
                          std::vector<double> const& costlist,
                          std::vector<int>& ranklist, int nranks);

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
                             std::vector<int>& ranklist, int nranks);

  static int AssignBlocksContigImproved(std::vector<double> const& costlist,
                                        std::vector<int>& ranklist, int nranks);

  static int AssignBlocksCppIter(std::vector<double> const& costlist,
                                 std::vector<int>& ranklist, int nranks);

  friend class LoadBalancingPoliciesTest;
  friend class PolicyTest;
  //  friend class PolicyExecCtx;
  //  friend class ScaleExecCtx;
};
}  // namespace amr
