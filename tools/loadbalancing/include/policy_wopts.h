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

std::unordered_map<std::string, const LBPolicyWithOpts> const kPolicyMap = {
    {"actual",
     {.id = "actual",
      .name = "Actual",
      .policy = LoadBalancePolicy::kPolicyActual}},
    {"baseline",
     {.id = "baseline",
      .name = "Baseline",
      .policy = LoadBalancePolicy::kPolicyContiguousUnitCost}},
    {"lpt",
     {.id = "lpt", .name = "LPT", .policy = LoadBalancePolicy::kPolicyLPT}},
    {"cdp",
     {.id = "cdp",
      .name = "Contiguous-DP (CDP)",
      .policy = LoadBalancePolicy::kPolicyContigImproved}},
    {"cdpi50",
     {.id = "cdpi50",
      .name = "CDP-I50",
      .policy = LoadBalancePolicy::kPolicyCppIter,
      .cdp_opts = {.niter_frac = 0, .niters = 50}}},
    {"cdpi250",
     {.id = "cdpi250",
      .name = "CDP-I250",
      .policy = LoadBalancePolicy::kPolicyCppIter,
      .cdp_opts = {.niter_frac = 0, .niters = 250}}},
    {"hybrid10",
     {.id = "hybrid10",
      .name = "Hybrid (10%)",
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = 0.1}}},
    {"hybrid20",
     {.id = "hybrid20",
      .name = "Hybrid (20%)",
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = 0.2}}},
    {"hybrid30",
     {.id = "hybrid30",
      .name = "Hybrid (30%)",
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = 0.3}}},
    {"hybrid50",
     {.id = "hybrid50",
      .name = "Hybrid (50%)",
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = 0.5}}},
    {"hybrid70",
     {.id = "hybrid70",
      .name = "Hybrid (70%)",
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = 0.7}}},
    {"hybrid90",
     {.id = "hybrid90",
      .name = "Hybrid (90%)",
      .policy = LoadBalancePolicy::kPolicyHybridCppFirstV2,
      .hcf_opts = {.v2 = true, .lpt_frac = 0.9}}},
};
}  // namespace amr
