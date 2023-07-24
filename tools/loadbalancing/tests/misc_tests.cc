//
// Created by Ankush J on 4/11/23.
//

#include "bin_readers.h"
#include "block_alloc_sim.h"
#include "policy_exec_ctx.h"
#include "prof_set_reader.h"

#include <gtest/gtest.h>
#include <pdlfs-common/env.h>

namespace amr {
class MiscTest : public ::testing::Test {
 protected:
  std::string GetLogPath(const char* output_dir, const char* policy_name) {
    return PolicyUtils::GetLogPath(output_dir, policy_name, ".csv");
  }

  void AssertApproxEqual(std::vector<double> const& a,
                         std::vector<double> const& b) {
    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); i++) {
      ASSERT_NEAR(a[i], b[i], 0.0001);
    }
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
  opts.prof_dir = "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini";
  opts.output_dir = opts.prof_dir + "/output";
  opts.env = pdlfs::Env::Default();

  BlockSimulator sim(opts);
  sim.Run();
}

TEST_F(MiscTest, prof_reader_test) {
  int rv;

  std::vector<std::string> all_profs = {
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini/"
      "prof.merged.evt0.csv",
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini/"
      "prof.merged.evt1.csv"};

  ProfileReader reader(all_profs[1].c_str());

  std::vector<int> times;
  int nlines_read;

  times.resize(0);
  rv = reader.ReadTimestep(-1, times, nlines_read);
  logf(LOG_DBUG, "RV: %d, Times: %s", rv, SerializeVector(times, 10).c_str());

  times.resize(0);
  rv = reader.ReadTimestep(0, times, nlines_read);
  logf(LOG_DBUG, "RV: %d, Times: %s", rv, SerializeVector(times, 10).c_str());

  times.resize(0);
  rv = reader.ReadTimestep(1, times, nlines_read);
  logf(LOG_DBUG, "RV: %d, Times: %s", rv, SerializeVector(times, 10).c_str());

  times.resize(0);
  rv = reader.ReadTimestep(2, times, nlines_read);
  logf(LOG_DBUG, "RV: %d, Times: %s", rv, SerializeVector(times, 10).c_str());

  times.resize(0);
  rv = reader.ReadTimestep(3, times, nlines_read);
  logf(LOG_DBUG, "RV: %d, Times: %s", rv, SerializeVector(times, 10).c_str());
}

TEST_F(MiscTest, prof_set_reader_test) {
  std::vector<std::string> all_profs = {
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini/"
      "prof.merged.evt0.csv",
      "/Users/schwifty/Repos/amr-data/20230424-prof-tags/ref-mini/"
      "prof.merged.evt1.csv"};

  ProfSetReader reader(all_profs);
  std::vector<int> times;
  int sub_ts = -1, rv;

  do {
    rv = reader.ReadTimestep(sub_ts, times);
    logf(LOG_DBUG, "[PSRTest] TS: %d, RV: %d, Times: %s", sub_ts, rv,
         SerializeVector(times, 10).c_str());
    sub_ts++;
  } while (rv);
}

TEST_F(MiscTest, ExtrapolateCosts1) {
  std::vector<double> costs_prev = {1.0};
  std::vector<int> refs = {0};
  std::vector<int> derefs = {};
  std::vector<double> costs_cur;

  PolicyUtils::ExtrapolateCosts(costs_prev, refs, derefs, costs_cur);
  logf(LOG_DBUG, "Costs Prev: %s", SerializeVector(costs_prev, 10).c_str());
  logf(LOG_DBUG, "Costs Cur: %s", SerializeVector(costs_cur, 10).c_str());
  AssertApproxEqual(costs_cur, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
}

TEST_F(MiscTest, ExtrapolateCosts2) {
  std::vector<double> costs_prev = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<int> refs = {};
  std::vector<int> derefs = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> costs_cur;

  PolicyUtils::ExtrapolateCosts(costs_prev, refs, derefs, costs_cur);
  logf(LOG_DBUG, "Costs Prev: %s", SerializeVector(costs_prev, 10).c_str());
  logf(LOG_DBUG, "Costs Cur: %s", SerializeVector(costs_cur, 10).c_str());
  AssertApproxEqual(costs_cur, {4.5});
}

TEST_F(MiscTest, ExtrapolateCosts3) {
  std::vector<double> costs_prev = {1.0, 2.0, 3.0, 4.0, 5.0,
                                    6.0, 7.0, 8.0, 9.0};
  std::vector<int> refs = {0};
  std::vector<int> derefs = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> costs_cur;

  PolicyUtils::ExtrapolateCosts(costs_prev, refs, derefs, costs_cur);
  logf(LOG_DBUG, "Costs Prev: %s", SerializeVector(costs_prev, 10).c_str());
  logf(LOG_DBUG, "Costs Cur: %s", SerializeVector(costs_cur, 10).c_str());
  AssertApproxEqual(costs_cur, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.5});
}
}  // namespace amr