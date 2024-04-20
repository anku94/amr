//
// Created by Ankush J on 4/28/23.
//

#pragma once

#include <vector>

#include "bin_readers.h"
#include "common.h"
#include "policy.h"
#include "policy_exec_ctx.h"
#include "policy_stats.h"
#include "prof_set_reader.h"
#include "trace_utils.h"

namespace amr {
struct BlockSimulatorOpts {
  int nblocks;
  int nranks;
  int nts;
  int nts_toskip;
  std::string prof_dir;
  std::string output_dir;
  pdlfs::Env* env;
  std::vector<int> events;
  const char* prof_time_combine_policy;
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
        prof_reader_(Utils::LocateTraceFiles(options_.env, options_.prof_dir,
                                             options_.events),
                     Utils::ParseProfTimeCombinePolicy(
                         options_.prof_time_combine_policy)),
        nblocks_next_expected_(-1),
        num_lb_(0) {}

  void SetupAllPolicies();

  void SetupPolicy(PolicyExecOpts& opts) {
    policies_.emplace_back(opts);
    stats_.emplace_back(opts);
  }

  void Run();

  int RunTimestep(int& ts, int sub_ts);

  int InvokePolicies(int sub_ts, std::vector<double> const& cost_oracle,
                     std::vector<int>& ranklist_actual, std::vector<int>& refs,
                     std::vector<int>& derefs);

 private:
  int ReadTimestepInternal(int ts, int sub_ts, std::vector<int>& refs,
                           std::vector<int>& derefs,
                           std::vector<int>& assignments,
                           std::vector<int>& times);

  void LogSummary();

  BlockSimulatorOpts const options_;

  RefinementReader ref_reader_;
  AssignmentReader assign_reader_;
  ProfSetReader prof_reader_;

  std::vector<int> ranklist_;
  int nblocks_next_expected_;
  int num_lb_;

  std::vector<PolicyExecCtx> policies_;
  std::vector<PolicyStats> stats_;
};
}  // namespace amr
