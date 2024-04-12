//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace amr {
struct PolicyOptsILP {
  float obj_lb_time_limit;
  float obj_lb_rel_tol;
  float obj_lb_mip_gap;

  float obj_loc_time_limit;
  float obj_loc_mip_gap;

  PolicyOptsILP()
      : obj_lb_time_limit(10),
        obj_lb_rel_tol(0.1),
        obj_lb_mip_gap(0.1),
        obj_loc_time_limit(10),
        obj_loc_mip_gap(0.1) {}

  std::string ToString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(0);
    ss << "\n\tobj_lb_time_limit: \t" << std::to_string(obj_lb_time_limit)
       << "\n\tobj_lb_rel_tol:\t" << std::to_string(obj_lb_rel_tol)
       << "\n\tobj_lb_mip_gap:\t" << std::to_string(obj_lb_mip_gap)
       << "\n\tobj_loc_time_limit: \t" << std::to_string(obj_loc_time_limit)
       << "\n\tobj_loc_mip_gap:\t" << std::to_string(obj_loc_mip_gap);
    return ss.str();
  }
};

struct PolicyOptsHybrid {
  float frac_lpt;
  float lpt_target;

  PolicyOptsHybrid() : frac_lpt(0.2), lpt_target(0) {}

  std::string ToString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "\n\tnum_lpt: \t" << std::to_string(frac_lpt);
    ss << "\n\tlpt_target: \t" << std::to_string(lpt_target);
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
                                 std::vector<int>& ranklist, int nranks,
                                 void* opts = nullptr);

  static int AssignBlocksHybrid(std::vector<double> const& costlist,
                                std::vector<int>& ranklist, int nranks);

  static int AssignBlocksHybridCppFirst(std::vector<double> const& costlist,
                                        std::vector<int>& ranklist, int nranks,
                                        bool v2);

  friend class LoadBalancingPoliciesTest;
  friend class PolicyTest;
  //  friend class PolicyExecCtx;
  //  friend class ScaleExecCtx;
};
}  // namespace amr
