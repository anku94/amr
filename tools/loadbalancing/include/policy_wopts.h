#pragma once

#include <iomanip>
#include <string>

#include "policy.h"

namespace amr {
struct PolicyOptsCDPI {
  double niter_frac;
  int niters;
};

struct PolicyOptsHybridCDPFirst {
  bool v2;              // whether to use v2 or not, always yes
  double lpt_frac;      // frac of ranks to run LPT on
  int alt_solncnt_max;  // max no. of alt solns to explore
};

struct PolicyOptsILP {
  float obj_lb_time_limit;
  float obj_lb_rel_tol;
  float obj_lb_mip_gap;

  float obj_loc_time_limit;
  float obj_loc_mip_gap;

  PolicyOptsILP()
      : obj_lb_time_limit(10)
      , obj_lb_rel_tol(0.1)
      , obj_lb_mip_gap(0.1)
      , obj_loc_time_limit(10)
      , obj_loc_mip_gap(0.1) {}

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

struct PolicyOptsChunked {
  int chunk_size;
  int parallelism;

  std::string ToString() const {
    return std::string("\n\tchunk_size: \t") + std::to_string(chunk_size) +
           std::string("\n\tparallelism: \t") + std::to_string(parallelism);
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
    PolicyOptsChunked chunked_opts;
  };
};
}  // namespace amr
