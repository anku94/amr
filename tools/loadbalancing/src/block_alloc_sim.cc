//
// Created by Ankush J on 5/1/23.
//

#include "block_alloc_sim.h"

namespace amr {

void BlockSimulator::Run(int nts) {
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
  for (sub_ts = 0; sub_ts < nts; sub_ts++) {
    int ts;
    int rv = RunTimestep(ts, sub_ts);
    if (rv == 0) break;
  }

  fort::char_table table;
  LogSummary(table);

  logf(LOG_INFO, "Simulation finished. Num TS: %d", sub_ts);
  logf(LOG_INFO, "Num LBs: %d", num_lb_);
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

  rv = prof_reader_.ReadTimestep(sub_ts - 1, times);
  logf(LOG_DBUG, "[BlockSim] [ProfSetReader] RV: %d, Times: %s", rv,
       SerializeVector(times, 10).c_str());
  if (times.size() != block_assignments.size()) {
    logf(LOG_WARN, "times.size() != block_assignments.size() (%d, %d)",
         times.size(), block_assignments.size());
    times.resize(block_assignments.size());
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

  int nblocks_cur = assignments.size();
  int ref_sz = refs.size(), deref_sz = derefs.size();

  UpdateExpectedBlockCount(ts, sub_ts, nblocks_cur, ref_sz, deref_sz);

  std::vector<double> costs(times.begin(), times.end());
  InvokePolicies(costs, refs, derefs);

  return 0;
}
}  // namespace amr
