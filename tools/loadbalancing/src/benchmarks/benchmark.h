#include "alias_method.h"
#include "benchmark_stats.h"
#include "common.h"
#include "distributions.h"
#include "lb_policies.h"
#include "policy.h"
#include "tabular_data.h"

#include <cstdio>

namespace amr {
struct BenchmarkOpts {
  int nranks;
  int nblocks;
  BenchmarkOpts() : nranks(0), nblocks(0) {}
};

class Benchmark {
 public:
  explicit Benchmark(BenchmarkOpts& opts) : opts_(opts) {}

  void Run() {
    logf(LOG_INFO, "Nranks: %d", opts_.nranks);
    RunPolicy();
  }

  void RunPolicy() {
    std::vector<double> costs;

    DistributionUtils::GenGaussian(costs, opts_.nblocks, 10.0, 0.5);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());
    //    EvaluatePolicySuite("Gaussian", costs);

    DistributionUtils::GenExponential(costs, opts_.nblocks, 1.0);
    logf(LOG_INFO, "Times: %s", SerializeVector(costs, 10).c_str());
    //    EvaluatePolicySuite("Exponential", costs);

    DistributionUtils::GenPowerLaw(costs, opts_.nblocks, -3.0, 50, 100);
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
        EvaluatePolicy(distrib_name, policy_name, p, nullptr, costs, rtavg, rtmax);
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
    std::vector<int> ranks(costs.size());
    LoadBalancePolicies::AssignBlocks(p, costs, ranks, opts_.nranks, p_opts);
    std::vector<double> rank_times;
    PolicyUtils::ComputePolicyCosts(opts_.nranks, costs, ranks, rank_times,
                                    time_avg, time_max);
    logf(LOG_INFO,
         "[%-20s] Placement evaluated. Avg Cost: %.2f, Max Cost: %.2f",
         policy_name.c_str(), time_avg, time_max);

    std::shared_ptr<Row> row = std::make_shared<BenchmarkRow>(
        opts_.nranks, opts_.nblocks, distrib_name, policy_name, time_avg,
        time_max);
    table_.addRow(row);
  }

 private:
  const BenchmarkOpts opts_;
  TabularData table_;
};
}  // namespace amr