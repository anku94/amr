//
// Created by Ankush J on 4/28/23.
//

#pragma once

#include "bin_readers.h"
#include "common.h"

#include <vector>

namespace amr {
struct BlockSimulatorOpts {
  int nblocks;
  int nranks;
  const char* trace_root;
};

#define FAIL_IF(cond, msg) \
  if (cond) {              \
    logf(LOG_ERRO, msg);   \
    ABORT(msg);            \
  }

class BlockSimulator {
 public:
  BlockSimulator(BlockSimulatorOpts const& opts)
      : opts_(opts),
        nblocks_next_expected(-1),
        ref_reader_(opts_.trace_root),
        assign_reader_(opts_.trace_root),
        ts_(0),
        sub_ts_(0) {}

  void Run(int nts = INT_MAX) {
    InitialAlloc();

    /* Semantics: every sub_ts exists in the assignment log, but
     * refinement log is sparse. A sub_ts not present in the assignment log
     * indicates corruption.
     * The code below sets the ts for the current sub_ts
     */
    for (int sub_ts = 0; sub_ts < nts; sub_ts++) {
      int ts;
      int rv = RunTimestep(ts, sub_ts);
      if (rv == 0) return;
    }
  }

  void InitialAlloc() {
    assert(opts_.nblocks % opts_.nranks == 0);
    int nblocks_per_rank = opts_.nblocks / opts_.nranks;

    nblocks_next_expected = opts_.nblocks;

    for (int i = 0; i < opts_.nranks; ++i) {
      for (int j = 0; j < nblocks_per_rank; ++j) {
        ranklist_.push_back(i);
      }
    }
  }

  int RunTimestep(int& ts, int sub_ts) {
    int rv;

    std::vector<int> block_assignments;
    std::vector<int> refs, derefs;

    logf(LOG_DBG2, "========================================");

    rv = assign_reader_.ReadTimestep(ts, sub_ts, block_assignments);
    FAIL_IF(rv < 0, "Error in AssRd/ReadTimestep");
    logf(LOG_DBG2, "[BlockSim] [AssRd] TS:%d_%d, rv: %d\nAssignments: %s", ts,
         sub_ts, rv, SerializeVector(block_assignments, 10).c_str());

    if (rv == 0) return 0;

    int ts_rr;
    rv = ref_reader_.ReadTimestep(ts_rr, sub_ts, refs, derefs);
    FAIL_IF(rv < 0, "Error in RefRd/ReadTimestep");
    logf(LOG_DBG2,
         "[BlockSim] [RefRd] TS:%d_%d, rv: %d\n\tRefs: %s\n\tDerefs: %s", ts,
         sub_ts, rv, SerializeVector(refs, 10).c_str(),
         SerializeVector(derefs, 10).c_str());

    if (ts_rr != -1) assert(ts_rr == ts);

    ReadTimestepInternal(ts, sub_ts, refs, derefs, block_assignments);

    return 1;
  }

 private:
  int ReadTimestepInternal(int ts, int sub_ts, const std::vector<int>& refs,
                           std::vector<int>& derefs,
                           std::vector<int>& assignments) {
    logf(LOG_DBG2, "----------------------------------------");

    int nblocks_cur = assignments.size();

    if (nblocks_next_expected != -1 && nblocks_next_expected != nblocks_cur) {
      logf(LOG_ERRO, "nblocks_next_expected != assignments.size()");
      ABORT("nblocks_next_expected != assignments.size()");
    }

    nblocks_next_expected = nblocks_cur;
    nblocks_next_expected += refs.size() * 7;
    nblocks_next_expected -= derefs.size() * 7 / 8;

    logf(LOG_DBUG, "[BlockSim] TS:%d_%d, nblocks: %d->%d", ts, sub_ts,
         nblocks_cur, nblocks_next_expected);

    return 0;
  }

  BlockSimulatorOpts const opts_;

  int nblocks_next_expected;
  std::vector<int> ranklist_;

  RefinementReader ref_reader_;
  AssignmentReader assign_reader_;

  int ts_;
  int sub_ts_;
};
}  // namespace amr