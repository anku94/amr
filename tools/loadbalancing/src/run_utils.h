#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "lb_policies.h"

namespace amr {
struct RunType {
  int nranks;
  int nblocks;
  std::string policy;

  std::string ToString() const {
    std::stringstream ss;
    ss << std::string(60, '-') << "\n";
    ss << "[BenchmarkRun] policy: " << policy << ", n_ranks: " << nranks
       << ", n_blocks: " << nblocks;
    return ss.str();
  }

  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks) {
    int rv = LoadBalancePolicies::AssignBlocks(policy.c_str(), costlist,
                                               ranklist, nranks);
    return rv;
  }
};
}  // namespace amr
