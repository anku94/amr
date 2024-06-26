#pragma once

#include "bench_util.h"
#include "benchmark_stats.h"
#include "common.h"
#include "config_parser.h"
#include "distrib/distributions.h"
#include "globals.h"
#include "lb_policies.h"
#include "policy_utils.h"
#include "run_utils.h"
#include "tabular_data.h"
#include "trace_utils.h"

#include <pdlfs-common/env.h>

namespace amr {
struct BenchmarkOpts {
  pdlfs::Env* const env;
  std::string config_file;
  std::string output_dir;
};

class Benchmark {
 public:
  explicit Benchmark(BenchmarkOpts& opts)
      : opts_(opts), utils_(opts_.output_dir, pdlfs::Env::Default()) {
    Globals.config = std::make_unique<ConfigParser>(opts_.config_file);
  }

  void Run() {
    // int nruns = RunSuiteMini();
    int nruns = RunSuite();
    std::string table_path = opts_.output_dir + "/benchmark.csv";
    Utils::EnsureDir(opts_.env, opts_.output_dir);
    EmitTable(table_path, nruns);
  }

  int RunSuiteMini() {
    int nranks = 64;
    int nblocks = 150;

    RunType base{nranks, nblocks, "baseline"};

    // RunType hybrid = base;
    // hybrid.policy = LoadBalancePolicy::kPolicyHybrid;
    //
    // RunType hybrid2 = base;
    // hybrid2.policy = LoadBalancePolicy::kPolicyHybridCppFirst;

    RunType hybrid3 = base;
    hybrid3.policy = "hybrid";

    RunType lpt = base;
    lpt.policy = "lpt";

    RunType cpp = base;
    cpp.policy = "cdp";

    RunType cpp_iter = base;
    cpp_iter.policy = "cdpi50";

    int iter = 50;
    cpp_iter.policy_opts = &iter;
    cpp_iter.policy_name = "CppIter_50";

    // std::vector<RunType> all_runs{base, cpp, cpp_iter, lpt, hybrid, hybrid2};
    std::vector<RunType> all_runs{ hybrid3, lpt};

    for (auto& r : all_runs) {
      logv(__LOG_ARGS__, LOG_INFO, "[RUN] %s", r.ToString().c_str());
    //   if (r.policy_opts && r.policy == LoadBalancePolicy::kPolicyHybrid) {
    //     auto* opts = reinterpret_cast<PolicyOptsHybrid*>(r.policy_opts);
    //     logv(__LOG_ARGS__, LOG_INFO, "%s", opts->ToString().c_str());
    //   }
    }

    DoRuns(all_runs);

    return all_runs.size();
  }

  int RunSuite() {
    std::vector<int> all_ranks = {512, 1024, 2048, 4096, 8192, 16384};
    // std::vector<int> all_ranks = {512};
    std::vector<int> all_blocks = {1000, 2000, 4000, 8000, 16000, 32000};

    int nruns = 0;

    for (auto r : all_ranks) {
      for (auto b : all_blocks) {
        if (b < r) continue;
        nruns = RunCppIterSuite(r, b);
      }
    }

    return nruns;
  }

  int RunCppIterSuite(int nranks, int nblocks) {
    logv(__LOG_ARGS__, LOG_INFO, "[CppIterSuite] nranks: %d, nblocks: %d", nranks, nblocks);

    // std::vector<int> all_iters = {1,   10,  50,   100,  250,
    //                               500, 750, 1000, 1250, 1500};
    std::vector<int> all_iters = {50, 250};

    RunType base{nranks, nblocks, "baseline"};

    RunType cpp = base;
    cpp.policy = "cdp";

    std::vector<RunType> all_runs{base, cpp};
    // std::vector<RunType> all_runs{lpt};

    for (int iter_idx = 0; iter_idx < all_iters.size(); iter_idx++) {
      std::string policy_id =
          std::string("cdpi") + std::to_string(all_iters[iter_idx]);
      RunType cpp_iter = base;
      cpp_iter.policy_opts = &all_iters[iter_idx];
      cpp_iter.policy = policy_id;
      cpp_iter.policy_name = policy_id;
      all_runs.push_back(cpp_iter);
    }

    RunType lpt = base;
    lpt.policy = "lpt";
    all_runs.push_back(lpt);

    // RunType hybrid = base;
    // hybrid.policy = "
    // all_runs.push_back(hybrid);
    //
    // RunType hybrid2 = base;
    // hybrid2.policy = LoadBalancePolicy::kPolicyHybridCppFirst;
    // all_runs.push_back(hybrid2);

    RunType hybrid3 = base;
    hybrid3.policy = "hybrid";
    all_runs.push_back(hybrid3);

    for (auto& r : all_runs) {
      logv(__LOG_ARGS__, LOG_INFO, "[RUN]\n\t%s", r.ToString().c_str());
    }

    DoRuns(all_runs);

    return all_runs.size();
  }

  //
  // Call DoRun() on run_vec. Generate costs for the first run in the vector
  // Same costs is passed to all runs.
  //
  void DoRuns(std::vector<RunType>& rvec) {
    if (rvec.empty()) return;

    std::vector<double> costs;
    auto& r0 = rvec[0];
    DistributionUtils::GenDistributionWithDefaults(costs, r0.nblocks);
    logv(__LOG_ARGS__, LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());

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
    logv(__LOG_ARGS__, LOG_INFO, "%s", r.ToString().c_str());

    std::vector<int> ranks(costs.size());
    LoadBalancePolicies::AssignBlocks(r.policy.c_str(), costs, ranks, r.nranks);
    std::vector<double> rank_times;
    PolicyUtils::ComputePolicyCosts(r.nranks, costs, ranks, rank_times,
                                    time_avg, time_max);
    logv(__LOG_ARGS__, LOG_INFO,
         "[%-20s] Placement evaluated. Avg Cost: %.2f, Max Cost: %.2f",
         r.policy_name.c_str(), time_avg, time_max);

    utils_.LogVector("Costs", costs);
    utils_.LogVector("Ranks", ranks);
    utils_.LogVector("Rank times", rank_times);
    utils_.LogAllocation(r.nblocks, r.nranks, costs, ranks);

    Distribution d = DistributionUtils::GetConfigDistribution();
    std::string distrib_name = DistributionUtils::DistributionToString(d);

    std::shared_ptr<TableRow> row = std::make_shared<BenchmarkRow>(
        r.nranks, r.nblocks, distrib_name, r.policy_name, time_avg, time_max);
    table_.addRow(row);

    auto pex_fpath =
        utils_.GetPexFileName(r.policy_name, distrib_name, r.nranks, r.nblocks);
    utils_.WritePexToFile(pex_fpath, costs, ranks, r.nranks);
  }

  void EmitTable(const std::string& table_out, int partition_intvl) {
    std::stringstream table_stream;
    table_.emitTable(table_stream, partition_intvl);
    logv(__LOG_ARGS__, LOG_INFO, "Table: \n%s", table_stream.str().c_str());

    utils_.WriteToFile(table_out, table_.toCSV());
  }

 private:
  const BenchmarkOpts opts_;
  TabularData table_;
  BenchmarkUtils utils_;
};
}  // namespace amr
