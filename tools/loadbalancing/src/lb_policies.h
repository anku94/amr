//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include "common.h"

#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>

namespace amr {
enum class LoadBalancingPolicy;
class LoadBalancePolicies {
 public:
  static int AssignBlocks(LoadBalancingPolicy policy, std::vector<double> const& costlist,
                           std::vector<int>& ranklist, int nranks) {
    // Two reasons to have a static object:
    // - once per lifetime lb_policy selection
    // - once-per-lifetime logging of lb_policy selection

    static LoadBalancePolicies lb_instance(policy);
    if (lb_instance.policy_ != policy) {
      ABORT("Only one lb_policy supported during program lifetime!");
    }

    return AssignBlocksInternal(lb_instance.policy_, costlist, ranklist, nranks);
  }

  LoadBalancePolicies(LoadBalancePolicies const&) = delete;
  void operator=(LoadBalancePolicies const&) = delete;

 private:
  LoadBalancePolicies(LoadBalancingPolicy policy) : policy_(policy) {
    std::string policy_str = PolicyToString(policy);
    logf(LOG_INFO, "[LoadBalancePolicies] Selected LoadBalancingPolicy: %s",
         policy_str.c_str());
  }

  static std::string PolicyToString(LoadBalancingPolicy policy);

  static int AssignBlocksInternal(LoadBalancingPolicy policy,
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

  const LoadBalancingPolicy policy_;

  friend class LoadBalancingPoliciesTest;
  friend class PolicyExecCtx;
};
}  // namespace amr