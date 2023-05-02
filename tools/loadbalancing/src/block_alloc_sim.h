//
// Created by Ankush J on 4/28/23.
//

#pragma once

#include "bin_readers.h"
#include "common.h"
#include "policy.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"
#include "utils.h"

#include <vector>

namespace amr {
struct BlockSimulatorOpts {
  int nblocks;
  int nranks;
  std::string prof_dir;
  std::string output_dir;
  pdlfs::Env* env;
};

#define FAIL_IF(cond, msg) \
  if (cond) {              \
    logf(LOG_ERRO, msg);   \
    ABORT(msg);            \
  }

class BlockSimulator {
 public:
  explicit BlockSimulator(BlockSimulatorOpts& opts)
      : options_(opts),
        ref_reader_(options_.prof_dir),
        assign_reader_(options_.prof_dir),
        prof_reader_(Utils::LocateTraceFiles(options_.env, options_.prof_dir)),
        nblocks_next_expected_(-1),
        num_lb_(0) {}

  void SetupAllPolicies() {
    policies_.clear();
    policies_.emplace_back(options_.output_dir.c_str(), "Contiguous/Unit-Cost",
                           Policy::kPolicyContiguous, options_.env,
                           options_.nranks, /* unit_cost */ true,
                           /* oracle_cost */ false);
    policies_.emplace_back(options_.output_dir.c_str(),
                           "Contiguous/Actual-Cost", Policy::kPolicyContiguous,
                           options_.env, options_.nranks, /* unit_cost */ false,
                           /* oracle_cost */ false);
    policies_.emplace_back(options_.output_dir.c_str(),
                           "RoundRobin/Actual-Cost", Policy::kPolicyRoundRobin,
                           options_.env, options_.nranks, /* unit_cost */ false,
                           /* oracle_cost */ false);
    policies_.emplace_back(options_.output_dir.c_str(), "LPT/Actual-Cost",
                           Policy::kPolicyLPT, options_.env,
                           options_.nranks, /* unit_cost */ false,
                           /* oracle_cost */ false);
    policies_.emplace_back(options_.output_dir.c_str(),
                           "LPT/Actual-Cost-Oracle", Policy::kPolicyLPT,
                           options_.env, options_.nranks, /* unit_cost */ false,
                           /* oracle_cost */ true);
  }

  int InvokePolicies(std::vector<double> const& cost_oracle,
                     std::vector<int>& refs, std::vector<int>& derefs) {
    int rv = 0;

    for (auto& policy : policies_) {
      rv = policy.ExecuteTimestep(cost_oracle, refs, derefs);
    }

    return rv;
  }

  void Run(int nts = INT_MAX);

  int RunTimestep(int& ts, int sub_ts);

 private:
  int ReadTimestepInternal(int ts, int sub_ts, std::vector<int>& refs,
                           std::vector<int>& derefs,
                           std::vector<int>& assignments,
                           std::vector<int>& times);

  void UpdateExpectedBlockCount(int ts, int sub_ts, int nblocks_cur, int ref_count,
                                int deref_count) {
    if (nblocks_next_expected_ != -1 && nblocks_next_expected_ != nblocks_cur) {
      logf(LOG_ERRO, "nblocks_next_expected_ != assignments.size()");
      ABORT("nblocks_next_expected_ != assignments.size()");
    }

    nblocks_next_expected_ = nblocks_cur;
    nblocks_next_expected_ += ref_count * 7;
    nblocks_next_expected_ -= deref_count * 7 / 8;

    logf(LOG_DBUG, "[BlockSim] TS:%d_%d, nblocks: %d->%d", ts, sub_ts,
         nblocks_cur, nblocks_next_expected_);
  }

  void LogSummary(fort::char_table& table) {
    table << fort::header << "Policy"
          << "Timesteps"
          << "ExcessCost"
          << "AvgCost"
          << "MaxCost"
          << "LocScore"
          << "ExecTime" << fort::endr;

    for (auto& policy : policies_) {
      policy.LogSummary(table);
    }

    logf(LOG_INFO, "\n%s", table.to_string().c_str());
  }

  BlockSimulatorOpts const options_;

  RefinementReader ref_reader_;
  AssignmentReader assign_reader_;
  ProfSetReader prof_reader_;

  std::vector<int> ranklist_;
  int nblocks_next_expected_;
  int num_lb_;

  std::vector<PolicyExecutionContext> policies_;
};
}  // namespace amr