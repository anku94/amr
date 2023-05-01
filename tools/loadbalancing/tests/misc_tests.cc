//
// Created by Ankush J on 4/11/23.
//

// #include "approx_pq.h"
#include "bin_readers.h"
#include "block_alloc_sim.h"
#include "policy_exec_ctx.h"

#include <gtest/gtest.h>

namespace amr {
class MiscTest : public ::testing::Test {
 protected:
  std::string GetLogPath(const char* output_dir, const char* policy_name) {
    return PolicyExecutionContext::GetLogPath(output_dir, policy_name);
  }
};

TEST_F(MiscTest, OutputFileTest) {
  std::string policy_name = "RoundRobin_Actual-Cost";
  std::string fname = GetLogPath("/a/b/c", policy_name.c_str());
  ASSERT_EQ(fname, "/a/b/c/roundrobin_actual_cost.csv");
}

TEST_F(MiscTest, RefReaderTest) {
  std::string fpath =
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini";
  RefinementReader reader(fpath);
  int ts, sub_ts = 0;
  int rv;
  std::vector<int> refs, derefs;

  do {
    rv = reader.ReadTimestep(ts, sub_ts, refs, derefs);
    logf(LOG_DBUG, "rv: %d, ts: %d, sub_ts: %d, refs: %s, derefs: %s", rv, ts,
         sub_ts, SerializeVector(refs, 10).c_str(),
         SerializeVector(derefs, 10).c_str());
    sub_ts++;

    if (sub_ts == 1000) break;
  } while (rv);
}

TEST_F(MiscTest, AssignReaderTest) {
  std::string fpath =
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini";
  AssignmentReader reader(fpath);
  int ts, sub_ts = 0;
  int rv;
  std::vector<int> blocks;

  do {
    rv = reader.ReadTimestep(ts, sub_ts, blocks);
    logf(LOG_DBUG, "rv: %d, ts: %d, sub_ts: %d, blocks: %s", rv, ts, sub_ts,
         SerializeVector(blocks, 10).c_str());
    sub_ts++;

    if (sub_ts == 1000) break;
  } while (rv);
}

TEST_F(MiscTest, BlockAllocSimTest) {
  BlockSimulatorOpts opts{};
  opts.nranks = 512;
  opts.nblocks = 512;
  opts.trace_root =
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini";

  BlockSimulator sim(opts);
  sim.Run();
}
}  // namespace amr