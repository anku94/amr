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

  LoadBalancePolicies(LoadBalancePolicies const&) = delete;
  void operator=(LoadBalancePolicies const&) = delete;

 private:
  explicit LoadBalancePolicies(LoadBalancePolicy policy);

  static int AssignBlocksInternal(LoadBalancePolicy policy,
                                  std::vector<double> const& costlist,
                                  std::vector<int>& ranklist, int nranks);

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

  const LoadBalancePolicy policy_;

  friend class LoadBalancingPoliciesTest;
  friend class PolicyExecCtx;
};
}  // namespace amr
