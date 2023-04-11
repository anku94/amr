//
// Created by Ankush J on 4/10/23.
//

#pragma once

#include "common.h"

#include <pdlfs-common/env.h>

namespace amr {
enum class Policy;
class PolicyExecutionContext {
 public:
  PolicyExecutionContext(const char* policy_name, Policy policy,
                         pdlfs::Env* env)
      : policy_name_(policy_name),
        policy_(policy),
        env_(env),
        ts_(0),
        excess_cost_(0),
        total_cost_avg_(0),
        total_cost_max_(0),
        exec_time_us_(0) {}

  /*
   * @param cost_alloc Cost-vector for assignment by policy
   * @param cost_actual Cost-vector for load balance estimation
   */
  int ExecuteTimestep(int nranks, std::vector<double> const& cost_alloc,
                       std::vector<double> const& cost_actual);

  /* cost is assumed to be us */
  void LogSummary() {
    logf(LOG_INFO, "Policy: %s (%d timesteps simulated)", policy_name_, ts_);
    logf(LOG_INFO, "-----------------------------------");
    logf(LOG_INFO, "\tExcess Cost: \t%.2f s", excess_cost_ / 1e6);
    logf(LOG_INFO, "\tAvg Cost: \t%.2f s", total_cost_avg_ / 1e6);
    logf(LOG_INFO, "\tMax Cost: \t%.2f s", total_cost_max_ / 1e6);

    logf(LOG_INFO, "\n\tExec Time: \t%.2f s\n", exec_time_us_ / 1e6);
  }

 private:
  const char* const policy_name_;
  const Policy policy_;
  pdlfs::Env* const env_;

  int ts_;

  /* cost is assumed to be us */
  double excess_cost_;
  double total_cost_avg_;
  double total_cost_max_;

  double exec_time_us_;
};
}  // namespace amr
