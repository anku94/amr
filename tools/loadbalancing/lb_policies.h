//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include "common.h"
#include "policy_sim.h"

#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>

namespace amr {
enum class Policy;
class LoadBalancePolicies {
 public:
  static void AssignBlocks(Policy policy, std::vector<double> const& costlist,
                           std::vector<int>& ranklist, int nranks) {
    // Two reasons to have a static object:
    // - once per lifetime policy selection
    // - once-per-lifetime logging of policy selection

    static LoadBalancePolicies lb_instance(policy);
    if (lb_instance.policy_ != policy) {
      ABORT("Only one policy supported during program lifetime!");
    }

    AssignBlocksInternal(lb_instance.policy_, costlist, ranklist, nranks);
  }

  LoadBalancePolicies(LoadBalancePolicies const&) = delete;
  void operator=(LoadBalancePolicies const&) = delete;

 private:
  LoadBalancePolicies(Policy policy) : policy_(policy) {
    std::string policy_str = PolicyToString(policy);
    logf(LOG_INFO, "[LoadBalancePolicies] Selected Policy: %s",
         policy_str.c_str());
  }

  static std::string PolicyToString(Policy policy);

  static void AssignBlocksInternal(Policy policy,
                                   std::vector<double> const& costlist,
                                   std::vector<int>& ranklist, int nranks);

  static void AssignBlocksRoundRobin(std::vector<double> const& costlist,
                                     std::vector<int>& ranklist, int nranks);

  static void AssignBlocksSkewed(std::vector<double> const& costlist,
                                 std::vector<int>& ranklist, int nranks);

  static void AssignBlocksContiguous(std::vector<double> const& costlist,
                                     std::vector<int>& ranklist, int nranks);

  static void AssignBlocksSPT(std::vector<double> const& costlist,
                              std::vector<int>& ranklist, int nranks);

  static void AssignBlocksLPT(std::vector<double> const& costlist,
                              std::vector<int>& ranklist, int nranks);

  const Policy policy_;

  friend class LoadBalancingPoliciesTest;
  friend class PolicyExecutionContext;
};
}  // namespace amr