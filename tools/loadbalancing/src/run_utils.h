#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "lb_policies.h"

namespace amr {
struct RunType {
  int nranks;
  int nblocks;
  std::string policy;
  void* policy_opts;
  std::string policy_name;

  std::string ToString() const {
    std::stringstream ss;
    std::string popts_str = (policy_opts != nullptr) ? "[set]" : "nullptr";
    std::string pname = !policy_name.empty()
                            ? policy_name
                            : policy;
    ss << std::string(60, '-') << "\n";
    ss << "[BenchmarkRun] n_ranks: " << nranks << ", n_blocks: " << nblocks
       << "\n\tpolicy: " << pname << ", opts: " << popts_str;
    return ss.str();
  }

  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks) {
    int rv = LoadBalancePolicies::AssignBlocks(policy.c_str(), costlist, ranklist,
                                               nranks);
    return rv;
  }
};

class RunSuites {
 public:
  static std::vector<RunType> GetSuiteMini(int nranks, int nblocks) {
    std::vector<RunType> suite;

    RunType base{nranks, nblocks, "baseline"};

    // RunType hybrid = base;
    // hybrid.policy = "hybrid";

    // RunType hybrid2 = base;
    // hybrid2.policy = LoadBalancePolicy::kPolicyHybridCppFirst;

    RunType hybrid3 = base;
    hybrid3.policy = "hybrid";

    suite.push_back(base);
    // suite.push_back(hybrid);
    // suite.push_back(hybrid2);
    suite.push_back(hybrid3);

    return suite;
  }

  static std::vector<RunType> GetCppIterSuite(int nblocks, int nranks) {
    std::vector<int> all_iters = {50, 250};

    RunType base{nranks, nblocks, "baseline"};

    RunType cpp = base;
    cpp.policy = "cdp";

    std::vector<RunType> all_runs{base, cpp};
    // std::vector<RunType> all_runs{lpt};

    for (int iter_idx = 0; iter_idx < all_iters.size(); iter_idx++) {
      auto policy_id = std::string("cdpi") + std::to_string(all_iters[iter_idx]);
      RunType cpp_iter = base;
      cpp_iter.policy_opts = &all_iters[iter_idx];
      cpp_iter.policy = policy_id.c_str();
      cpp_iter.policy_name = policy_id;
      all_runs.push_back(cpp_iter);
    }

    RunType lpt = base;
    lpt.policy = "lpt";
    all_runs.push_back(lpt);

    // RunType hybrid = base;
    // hybrid.policy = LoadBalancePolicy::kPolicyHybrid;
    // all_runs.push_back(hybrid);
    //
    // RunType hybrid2 = base;
    // hybrid2.policy = LoadBalancePolicy::kPolicyHybridCppFirst;
    // all_runs.push_back(hybrid2);

    RunType hybrid3 = base;
    hybrid3.policy = "hybrid";
    all_runs.push_back(hybrid3);

    return all_runs;
  }
};

}  // namespace amr
