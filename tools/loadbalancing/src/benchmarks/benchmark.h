#pragma once

#include "bench_util.h"
#include "benchmark_stats.h"
#include "common.h"
#include "config_parser.h"
#include "globals.h"
#include "distributions.h"
#include "lb_policies.h"
#include "policy.h"
#include "tabular_data.h"
#include "trace_utils.h"

#include <pdlfs-common/env.h>

namespace amr {
struct BenchmarkOpts {
  pdlfs::Env* const env;
  std::string config_file;
  std::string output_dir;
};

struct RunType {
  int nranks;
  int nblocks;
  Distribution d;
  LoadBalancePolicy policy;
  void* policy_opts;
  std::string policy_name;

  std::string ToString() const {
    std::stringstream ss;
    std::string dname = DistributionUtils::DistributionToString(d);
    std::string popts_str = (policy_opts != nullptr) ? "[set]" : "nullptr";
    std::string pname = !policy_name.empty()
                            ? policy_name
                            : PolicyUtils::PolicyToString(policy);
    ss << std::string(60, '-') << "\n";
    ss << "[BenchmarkRun] n_ranks: " << nranks << ", n_blocks: " << nblocks
       << "\n\t \tdistrib: " << dname << ", policy: " << pname
       << ", opts: " << popts_str;
    return ss.str();
  }
};

class Benchmark {
 public:
  explicit Benchmark(BenchmarkOpts& opts)
      : opts_(opts), utils_(opts_.output_dir, pdlfs::Env::Default()) {
        Globals.config = std::make_unique<ConfigParser>(opts_.config_file);
  }

  void Run() {
    RunSuiteMini();
    // RunSuite();
    std::string table_path = opts_.output_dir + "/benchmark.csv";
    Utils::EnsureDir(opts_.env, opts_.output_dir);
    EmitTable(table_path);
  }

  void RunSuiteMini() {
    int nranks = 64;
    int nblocks = 200;
    RunType base{nranks, nblocks, Distribution::kPowerLaw,
                 LoadBalancePolicy::kPolicyContiguousUnitCost};

    RunType hybrid = base;
    hybrid.policy = LoadBalancePolicy::kPolicyHybrid;

    // RunType lpt = base;
    // lpt.policy = LoadBalancePolicy::kPolicyLPT;
    //
    // RunType cpp = base;
    // cpp.policy = LoadBalancePolicy::kPolicyContigImproved;
    //
    // RunType cpp_iter = base;
    // cpp_iter.policy = LoadBalancePolicy::kPolicyCppIter;
    // int iter = 50;
    // cpp_iter.policy_opts = &iter;
    // cpp_iter.policy_name = "CppIter_50";

    // std::vector<RunType> all_runs{base, hybrid, lpt, cpp, cpp_iter};
    std::vector<RunType> all_runs{hybrid};

    for (auto& r : all_runs) {
      logf(LOG_INFO, "[RUN] %s", r.ToString().c_str());
      if (r.policy_opts && r.policy == LoadBalancePolicy::kPolicyHybrid) {
        auto* opts = reinterpret_cast<PolicyOptsHybrid*>(r.policy_opts);
        logf(LOG_INFO, "%s", opts->ToString().c_str());
      }
    }

    DoRuns(all_runs);
  }

  void RunSuite() {
    std::vector<int> all_ranks = {512, 1024, 2048, 4096, 8192, 16384};
    // std::vector<int> all_ranks = {512};
    std::vector<int> all_blocks = {1000, 2000, 4000, 8000, 16000, 32000};

    for (auto r : all_ranks) {
      for (auto b : all_blocks) {
        if (b < r) continue;
        RunCppIterSuite(r, b);
      }
    }
  }

  void RunCppIterSuite(int nranks, int nblocks) {
    logf(LOG_INFO, "[CppIterSuite] nranks: %d, nblocks: %d", nranks, nblocks);

    // std::vector<int> all_iters = {1,   10,  50,   100,  250,
    //                               500, 750, 1000, 1250, 1500};
    std::vector<int> all_iters = {50, 250};

    RunType base{nranks, nblocks, Distribution::kPowerLaw,
                 LoadBalancePolicy::kPolicyContiguousUnitCost};
    RunType lpt = base;
    lpt.policy = LoadBalancePolicy::kPolicyLPT;
    RunType cpp = base;
    cpp.policy = LoadBalancePolicy::kPolicyContigImproved;

    RunType hybrid = base;
    hybrid.policy = LoadBalancePolicy::kPolicyHybrid;

    std::vector<RunType> all_runs{base, lpt, cpp, hybrid};
    // std::vector<RunType> all_runs{lpt};

    for (int iter_idx = 0; iter_idx < all_iters.size(); iter_idx++) {
      RunType cpp_iter = base;
      cpp_iter.policy_opts = &all_iters[iter_idx];
      cpp_iter.policy = LoadBalancePolicy::kPolicyCppIter;
      cpp_iter.policy_name = PolicyUtils::PolicyToString(cpp_iter.policy) +
                             "_" + std::to_string(all_iters[iter_idx]);
      all_runs.push_back(cpp_iter);
    }

    for (auto& r : all_runs) {
      logf(LOG_INFO, "[RUN]\n\t%s", r.ToString().c_str());
    }

    DoRuns(all_runs);
  }

  //
  // Call DoRun() on run_vec. Generate costs for the first run in the vector
  // Same costs is passed to all runs.
  //
  void DoRuns(std::vector<RunType>& rvec) {
    if (rvec.empty()) return;

    std::vector<double> costs;
    auto& r0 = rvec[0];
    DistributionUtils::GenDistributionWithDefaults(r0.d, costs, r0.nblocks);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());

    for (auto& r : rvec) {
      DoRun(r, costs);
    }
  }

  //
  // Invoke a run, add results to table_
  //
  void DoRun(const RunType& r, std::vector<double> const& costs) {
    // Assume nranks, nblocks, d, policy, policy_opts are defined
    // Policy_name may be overwritten

    double time_avg, time_max;
    logf(LOG_INFO, "%s", r.ToString().c_str());

    std::vector<int> ranks(costs.size());
    LoadBalancePolicies::AssignBlocks(r.policy, costs, ranks, r.nranks,
                                      r.policy_opts);
    std::vector<double> rank_times;
    PolicyUtils::ComputePolicyCosts(r.nranks, costs, ranks, rank_times,
                                    time_avg, time_max);
    logf(LOG_INFO,
         "[%-20s] Placement evaluated. Avg Cost: %.2f, Max Cost: %.2f",
         r.policy_name.c_str(), time_avg, time_max);

    utils_.LogVector("Costs", costs);
    utils_.LogVector("Ranks", ranks);
    utils_.LogVector("Rank times", rank_times);
    utils_.LogAllocation(r.nblocks, r.nranks, costs, ranks);

    std::string distrib_name = DistributionUtils::DistributionToString(r.d);
    std::string policy_name = r.policy_name.empty()
                                  ? PolicyUtils::PolicyToString(r.policy)
                                  : r.policy_name;

    std::shared_ptr<TableRow> row = std::make_shared<BenchmarkRow>(
        r.nranks, r.nblocks, distrib_name, policy_name, time_avg, time_max);
    table_.addRow(row);

    auto pex_fpath =
        utils_.GetPexFileName(policy_name, distrib_name, r.nranks, r.nblocks);
    utils_.WritePexToFile(pex_fpath, costs, ranks, r.nranks);
  }

  void EmitTable(const std::string& table_out) {
    std::stringstream table_stream;
    table_.emitTable(table_stream, 6);
    logf(LOG_INFO, "Table: \n%s", table_stream.str().c_str());

    utils_.WriteToFile(table_out, table_.toCSV());
  }

 private:
  const BenchmarkOpts opts_;
  TabularData table_;
  BenchmarkUtils utils_;
};
}  // namespace amr
