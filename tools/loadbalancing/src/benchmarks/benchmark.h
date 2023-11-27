#include "alias_method.h"
#include "benchmark_stats.h"
#include "common.h"
#include "distributions.h"
#include "lb_policies.h"
#include "policy.h"
#include "tabular_data.h"
#include "trace_utils.h"

#include <cstdio>
#include <pdlfs-common/env.h>

namespace amr {
struct BenchmarkOpts {
  pdlfs::Env* const env;
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
    ss << "[BenchmarkRun] n_ranks: " << nranks << ", n_blocks: " << nblocks
       << "\n\t \tdistrib: " << dname << ", policy: " << pname
       << ", opts: " << popts_str;
    return ss.str();
  }
};

class Benchmark {
 public:
  explicit Benchmark(BenchmarkOpts& opts) : opts_(opts) {}

  void Run() {
//    RunSuiteMini();
     RunSuite();
    std::string table_path = opts_.output_dir + "/benchmark.csv";
    Utils::EnsureDir(opts_.env, opts_.output_dir);
    EmitTable(table_path);
  }

//  void RunSuiteMini() { RunCppIterSuite(4096, 12000); }

  void RunSuite() {
        std::vector<int> all_ranks = {512, 1024, 2048, 4096, 8192, 16384};
//    std::vector<int> all_ranks = {512, 1024, 2048};
    std::vector<int> all_blocks = {1000, 2000, 4000, 8000, 16000, 32000};

    for (auto r : all_ranks) {
      for (auto b : all_blocks) {
        if (b < r) continue;
        RunCppIterSuite(r, b);
      }
    }

    std::string table_path = opts_.output_dir + "/benchmark.csv";
    Utils::EnsureDir(opts_.env, opts_.output_dir);
    EmitTable(table_path);
  }

  void RunCppIterSuite(int nranks, int nblocks) {
    logf(LOG_INFO, "[CppIterSuite] nranks: %d, nblocks: %d", nranks, nblocks);

    std::vector<int> all_iters = {1,   10,  50,   100,  250,
                                  500, 750, 1000, 1250, 1500};

    RunType base{nranks, nblocks, Distribution::kPowerLaw,
                 LoadBalancePolicy::kPolicyContiguousUnitCost};
    RunType lpt = base;
    lpt.policy = LoadBalancePolicy::kPolicyLPT;
    RunType cpp = base;
    cpp.policy = LoadBalancePolicy::kPolicyContigImproved;

    std::vector<RunType> all_runs{base, lpt, cpp};

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
  // We only generate costs once, and use common costs for the entire suite
  //
  void DoRuns(std::vector<RunType>& rvec) {
    if (rvec.empty()) return;

    std::vector<double> costs;
    auto& r0 = rvec[0];
    DistributionUtils::GenDistribution(r0.d, costs, r0.nblocks);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());

    for (auto& r : rvec) {
      DoRun(r, costs);
    }
  }

  void DoRun(const RunType& r, std::vector<double> const& costs) {
    // Assume nranks, nblocks, d, policy, policy_opts are defined
    // Policy_name may be overwritten
    //    std::vector<double> costs;
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

    std::string distrib_name = DistributionUtils::DistributionToString(r.d);
    std::string policy_name = r.policy_name.empty()
                                  ? PolicyUtils::PolicyToString(r.policy)
                                  : r.policy_name;

    std::shared_ptr<TableRow> row = std::make_shared<BenchmarkRow>(
        r.nranks, r.nblocks, distrib_name, policy_name, time_avg, time_max);
    table_.addRow(row);
  }

  void EmitTable(const std::string& table_out) {
    std::stringstream table_stream;
    table_.emitTable(table_stream);
    logf(LOG_INFO, "Table: \n%s", table_stream.str().c_str());

    auto env = pdlfs::Env::Default();
    pdlfs::WritableFile* fh;
    pdlfs::Status s = env->NewWritableFile(table_out.c_str(), &fh);
    assert(s.ok());
    fh->Append(table_.toCSV().c_str());
    fh->Close();
  }

  void RunPolicy() {
    std::vector<double> costs;
    int nblocks = 2000;
    int nranks = 512;

    DistributionUtils::GenGaussian(costs, nblocks, 10.0, 0.5);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());
    //    EvaluatePolicySuite("Gaussian", costs);

    DistributionUtils::GenExponential(costs, nblocks, 1.0);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());
    //    EvaluatePolicySuite("Exponential", costs);

    DistributionUtils::GenPowerLaw(costs, nblocks, -3.0, 50, 100);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());
    EvaluatePolicySuite("PowerLaw", costs);
    EvaluateCppIterPolicySuite("PowerLaw", costs);

    std::stringstream table_stream;
    table_.emitTable(table_stream);
    logf(LOG_INFO, "Table: \n%s", table_stream.str().c_str());
  }

  void EvaluatePolicySuite(std::string distrib_name,
                           std::vector<double> costs) {
    double rtavg, rtmax;
    std::vector<LoadBalancePolicy> policies = {
        LoadBalancePolicy::kPolicyContiguousUnitCost,
        LoadBalancePolicy::kPolicyContiguousActualCost,
        LoadBalancePolicy::kPolicyRoundRobin, LoadBalancePolicy::kPolicySPT,
        LoadBalancePolicy::kPolicyLPT,
        //        LoadBalancePolicy::kPolicyILP,
        LoadBalancePolicy::kPolicyContigImproved,
        //          LoadBalancePolicy::kPolicyCppIter
    };

    for (auto p : policies) {
      std::string policy_name = PolicyUtils::PolicyToString(p);
      EvaluatePolicy(distrib_name, policy_name, p, nullptr, costs, rtavg,
                     rtmax);
    }
  }

  void EvaluateCppIterPolicySuite(std::string distrib_name,
                                  std::vector<double> costs) {
    double rtavg, rtmax;
    std::vector<int> all_iters = {1,   10,  50,   100,  250,
                                  500, 750, 1000, 1250, 1500};
    auto policy = LoadBalancePolicy::kPolicyCppIter;
    for (auto iter : all_iters) {
      std::string policy_name =
          PolicyUtils::PolicyToString(policy) + "_" + std::to_string(iter);
      EvaluatePolicy(distrib_name, policy_name, policy, &iter, costs, rtavg,
                     rtmax);
    }
  }

  void EvaluatePolicy(const std::string& distrib_name,
                      const std::string& policy_name, LoadBalancePolicy p,
                      void* p_opts, std::vector<double> const& costs,
                      double& time_avg, double& time_max) {
    //    std::vector<int> ranks(costs.size());
    //    LoadBalancePolicies::AssignBlocks(p, costs, ranks, opts_.nranks,
    //    p_opts); std::vector<double> rank_times;
    //    PolicyUtils::ComputePolicyCosts(opts_.nranks, costs, ranks,
    //    rank_times,
    //                                    time_avg, time_max);
    //    logf(LOG_INFO,
    //         "[%-20s] Placement evaluated. Avg Cost: %.2f, Max Cost: %.2f",
    //         policy_name.c_str(), time_avg, time_max);
    //
    //    std::shared_ptr<TableRow> row = std::make_shared<BenchmarkRow>(
    //        opts_.nranks, opts_.nblocks, distrib_name, policy_name,
    //        time_avg, time_max);
    //    table_.addRow(row);
  }

 private:
  const BenchmarkOpts opts_;
  TabularData table_;
};
}  // namespace amr