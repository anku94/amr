//
// Created by Ankush J on 5/1/23.
//

#include "block_alloc_sim.h"

namespace amr {
void BlockSimulator::SetupAllPolicies() {
  policies_.clear();

  PolicyExecOpts policy_opts;
  policy_opts.output_dir = options_.output_dir.c_str();
  policy_opts.env = options_.env;
  policy_opts.nranks = options_.nranks;
  policy_opts.nblocks_init = options_.nblocks;
  // XXX: hardcoded for now
  policy_opts.trigger_interval = 1000;
  logf(LOG_INFO, "Hardcoded trigger interval: %d\n",
       policy_opts.trigger_interval);

  policy_opts.SetPolicy("Actual/Actual-Cost", "actual",
                        CostEstimationPolicy::kExtrapolatedCost,
                        TriggerPolicy::kEveryNTimesteps);
  SetupPolicy(policy_opts);

  policy_opts.SetPolicy("LPT/Extrapolated-Cost", "lpt",
                        CostEstimationPolicy::kExtrapolatedCost,
                        TriggerPolicy::kEveryNTimesteps);
  SetupPolicy(policy_opts);

  // policy_opts.SetPolicy("kContigImproved/Extrapolated-Cost", "cdp",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("CppIter/Extrapolated-Cost", "cdpi50",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid10",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid20",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid30",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid50",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid70",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
  //
  // policy_opts.SetPolicy("Hybrid/Extrapolated-Cost", "hybrid90",
  //                       CostEstimationPolicy::kExtrapolatedCost,
  //                       TriggerPolicy::kEveryNTimesteps);
  // SetupPolicy(policy_opts);
}

void BlockSimulator::Run() {
  logf(LOG_INFO, "Using prof dir: %s", options_.prof_dir.c_str());
  logf(LOG_INFO, "Using output dir: %s", options_.output_dir.c_str());

  Utils::EnsureDir(options_.env, options_.output_dir);

  SetupAllPolicies();

  /* Semantics: every sub_ts exists in the assignment log, but
   * refinement log is sparse. A sub_ts not present in the assignment log
   * indicates corruption.
   * The code below sets the ts for the current sub_ts
   */
  int sub_ts;
  for (sub_ts = 0; sub_ts < options_.nts; sub_ts++) {
    int ts;
    int rv = RunTimestep(ts, sub_ts);
    if (rv == 0) break;
  }

  LogSummary();

  logf(LOG_INFO, "Simulation finished. Sub-timesteps simulated: %d.", sub_ts);
}

int BlockSimulator::RunTimestep(int& ts, int sub_ts) {
  int rv;

  std::vector<int> block_assignments;
  std::vector<int> refs, derefs;
  std::vector<int> times;

  logf(LOG_DBUG, "========================================");

  rv = assign_reader_.ReadTimestep(ts, sub_ts, block_assignments);
  FAIL_IF(rv < 0, "Error in AssRd/ReadTimestep");
  logf(LOG_DBUG, "[BlockSim] [AssRd] TS:%d_%d, rv: %d\nAssignments: %s", ts,
       sub_ts, rv, SerializeVector(block_assignments, 10).c_str());

  if (rv == 0) return 0;

  int ts_rr;
  rv = ref_reader_.ReadTimestep(ts_rr, sub_ts, refs, derefs);
  FAIL_IF(rv < 0, "Error in RefRd/ReadTimestep");
  logf(LOG_DBUG,
       "[BlockSim] [RefRd] TS:%d_%d, rv: %d\n\tRefs: %s\n\tDerefs: %s", ts,
       sub_ts, rv, SerializeVector(refs, 10).c_str(),
       SerializeVector(derefs, 10).c_str());

  if (ts_rr != -1) assert(ts_rr == ts);

  // XXX: commented out on 20240129, stochsg/mat.bin follows different
  // convention? rv = prof_reader_.ReadTimestep(sub_ts - 1, times);
  rv = prof_reader_.ReadTimestep(sub_ts, times);
  logf(LOG_DBUG, "[BlockSim] [ProfSetReader] RV: %d, Times: %s", rv,
       SerializeVector(times, 10).c_str());
  if (times.size() != block_assignments.size()) {
    logf(LOG_WARN,
         "[ts%d/%d] times.size() != block_assignments.size() (%d, %d)", sub_ts,
         ts, times.size(), block_assignments.size());
    times.resize(block_assignments.size(), 1);
  }

  ReadTimestepInternal(ts, sub_ts, refs, derefs, block_assignments, times);

  return 1;
}

int BlockSimulator::ReadTimestepInternal(int ts, int sub_ts,
                                         std::vector<int>& refs,
                                         std::vector<int>& derefs,
                                         std::vector<int>& assignments,
                                         std::vector<int>& times) {
  logf(LOG_DBG2, "----------------------------------------");

  if (nblocks_next_expected_ != -1 &&
      nblocks_next_expected_ != assignments.size()) {
    logf(LOG_ERRO, "nblocks_next_expected_ != assignments.size()");
    ABORT("nblocks_next_expected_ != assignments.size()");
  }

  nblocks_next_expected_ = PolicyExecCtx::GetNumBlocksNext(
      assignments.size(), refs.size(), derefs.size());

  logf(LOG_DBUG, "[BlockSim] TS:%d_%d, nblocks: %d->%d", ts, sub_ts,
       (int)assignments.size(), nblocks_next_expected_);

  std::vector<double> costs(times.begin(), times.end());
  InvokePolicies(sub_ts, costs, assignments, refs, derefs);

  return 0;
}

int BlockSimulator::InvokePolicies(int sub_ts,
                                   std::vector<double> const& cost_oracle,
                                   std::vector<int>& ranklist_actual,
                                   std::vector<int>& refs,
                                   std::vector<int>& derefs) {
  int rv = 0;

  int npolicies = policies_.size();

  for (int pidx = 0; pidx < npolicies; ++pidx) {
    auto& policy = policies_[pidx];
    double exec_time = 0;
    rv = policy.ExecuteTimestep(cost_oracle, ranklist_actual, refs, derefs,
                                exec_time);
    if (rv != 0) break;

    if (sub_ts >= options_.nts_toskip) {
      if (policy.IsActualPolicy()) {
        stats_[pidx].LogTimestep(cost_oracle, ranklist_actual, exec_time);
      } else {
        stats_[pidx].LogTimestep(cost_oracle, policy.GetRanklist(), exec_time);
      }
    }
  }

  return rv;
}

void BlockSimulator::LogSummary() {
  TabularData table;
  int n = stats_.size();

  for (int i = 0; i < n; ++i) {
    auto& stat = stats_[i];
    auto& policy = policies_[i];
    int ts_succeeded, ts_invoked;

    policy.GetTimestepCount(ts_succeeded, ts_invoked);
    auto row = stat.GetTableRow(ts_succeeded, ts_invoked);

    table.addRow(row);
  }

  std::stringstream ss;
  table.emitTable(ss);

  logf(LOG_INFO, "\n%s", ss.str().c_str());
}
}  // namespace amr
