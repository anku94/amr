//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace amr {
struct PolicyOptsILP {
  float mip_gap;
  float obj_lb_time_limit;
  float obj_loc_time_limit;
  float obj_lb_rel_gap;

  PolicyOptsILP()
      : mip_gap(0.1),
        obj_lb_time_limit(10),
        obj_loc_time_limit(90),
        obj_lb_rel_gap(0.2) {}

  std::string ToString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(0);
    ss << "\n\tmip_gap: \t" << std::to_string(mip_gap)
       << "\tobj_lb_time_limit: \t" << std::to_string(obj_lb_time_limit)
       << "\n\tobj_loc_time_limit: \t" << std::to_string(obj_loc_time_limit)
       << "\n\tobj_lb_rel_gap:\t" << std::to_string(obj_lb_rel_gap);

    return ss.str();
  }
};

enum class LoadBalancePolicy;
class LoadBalancePolicies {
 public:
  static int AssignBlocks(LoadBalancePolicy policy,
                          std::vector<double> const& costlist,
                          std::vector<int>& ranklist, int nranks,
                          void* opts = nullptr);

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
                             void* opts = nullptr);

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
