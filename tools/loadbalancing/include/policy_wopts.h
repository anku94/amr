#pragma once

#include "policy.h"

#include <iomanip>
#include <unordered_map>

namespace amr {
struct PolicyOptsCDPI {
  double niter_frac;
  int niters;
};

struct PolicyOptsHybridCDPFirst {
  bool v2;
  double lpt_frac;
};

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

struct LBPolicyWithOpts {
  std::string id;
  std::string name;
  LoadBalancePolicy policy;
  union {
    PolicyOptsCDPI cdp_opts;
    PolicyOptsHybridCDPFirst hcf_opts;
    PolicyOptsILP ilp_opts;
    PolicyOptsHybrid hybrid_opts;
  };
};
}  // namespace amr
