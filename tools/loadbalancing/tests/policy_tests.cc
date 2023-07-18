//
// Created by Ankush J on 5/8/23.
//

#include "common.h"
#include "iterative/solver.h"
#include "lb_policies.h"

#include <gtest/gtest.h>
#include <vector>

namespace amr {
class PolicyTest : public ::testing::Test {
 public:
  static int AssignBlocksContiguous(std::vector<double> const& costlist,
                                    std::vector<int>& ranklist, int nranks) {
    return LoadBalancePolicies::AssignBlocksContiguous(costlist, ranklist,
                                                       nranks);
  }

  static int AssignBlocksLPT(std::vector<double> const& costlist,
                             std::vector<int>& ranklist, int nranks) {
    return LoadBalancePolicies::AssignBlocksLPT(costlist, ranklist, nranks);
  }

  static int AssignBlocksContigImproved(std::vector<double> const& costlist,
                                        std::vector<int>& ranklist,
                                        int nranks) {
    return LoadBalancePolicies::AssignBlocksContigImproved(costlist, ranklist,
                                                           nranks);
  }

  testing::AssertionResult AssertAllRanksAssigned(
      std::vector<int> const& ranklist, int nranks) {
    std::vector<int> allocs(nranks, 0);

    for (auto rank : ranklist) {
      if (!(rank >= 0 and rank < nranks)) {
        return testing::AssertionFailure()
               << "Rank " << rank << " is not in range [0, " << nranks << ")";
      }
      allocs[rank]++;
    }

    for (int alloc_idx = 0; alloc_idx < nranks; alloc_idx++) {
      int alloc = allocs[alloc_idx];
      if (alloc < 1) {
        return testing::AssertionFailure()
               << "Rank " << alloc_idx << " has no blocks assigned";
      }
    }

    return testing::AssertionSuccess();
  }
};

TEST_F(PolicyTest, ContiguousTest1) {
#include "lb_test1.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksContiguous(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_FALSE(AssertAllRanksAssigned(ranklist, nranks));
}

TEST_F(PolicyTest, LPTTest1) {
#include "lb_test1.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksLPT(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));
}

TEST_F(PolicyTest, ContigImprovedTest1) {
  std::vector<double> costlist = {1, 2, 3, 2, 1};
  int nranks = 3;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));
  ASSERT_TRUE(ranklist[0] == 0);
  ASSERT_TRUE(ranklist[1] == 0);
  ASSERT_TRUE(ranklist[2] == 1);
  ASSERT_TRUE(ranklist[3] == 2);
  ASSERT_TRUE(ranklist[4] == 2);
}

TEST_F(PolicyTest, ContigImprovedTest2) {
#include "lb_test1.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));
}

TEST_F(PolicyTest, ContigImprovedTest3) {
#include "lb_test2.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));
}

TEST_F(PolicyTest, IterTest) {
#include "lb_test3.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));

  auto solver = Solver();
  solver.AssignBlocks(costlist, ranklist, nranks, 100);

  double avg_cost, max_cost;
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost);
  logf(LOG_DBUG, "IterativeSolver. Avg Cost: %.0lf, Max Cost: %.0lf\n",
       avg_cost, max_cost);
}

TEST_F(PolicyTest, IterTest2) {
#include "lb_test4.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksLPT(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));

  double avg_cost, max_cost_lpt, max_cost_cpp, max_cost_iter;
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_lpt);
  logf(LOG_INFO, "LPT. Avg Cost: %.0lf, Max Cost: %.0lf\n", avg_cost,
       max_cost_lpt);

  rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);
  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));

  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_cpp);
  logf(LOG_INFO, "Contig++. Avg Cost: %.0lf, Max Cost: %.0lf\n", avg_cost,
       max_cost_cpp);

  auto solver = Solver();
  int iters;
  solver.AssignBlocks(costlist, ranklist, nranks, max_cost_lpt, iters);
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost_iter);

  logf(LOG_INFO, "IterativeSolver finished. Took %d iters.", iters);
  logf(LOG_INFO,
       "Initial Cost: %.0lf, Target Cost: %.0lf.\n"
       "\t- Avg Cost: %.0lf, Max Cost: %.0lf",
       max_cost_cpp, max_cost_lpt, avg_cost, max_cost_iter);
}

TEST_F(PolicyTest, IterTest3) {
#include "lb_test4.h"
  logf(LOG_INFO, "Costlist Size: %zu\n", costlist.size());
  int nranks = 512;
  std::vector<int> ranklist(costlist.size(), -1);

  int rv = AssignBlocksContigImproved(costlist, ranklist, nranks);
  ASSERT_EQ(rv, 0);

  EXPECT_TRUE(AssertAllRanksAssigned(ranklist, nranks));

  auto solver = Solver();
  solver.AssignBlocks(costlist, ranklist, nranks, 100);

  double avg_cost, max_cost;
  Solver::AnalyzePlacement(costlist, ranklist, nranks, avg_cost, max_cost);
  logf(LOG_DBUG, "IterativeSolver. Avg Cost: %.0lf, Max Cost: %.0lf\n",
       avg_cost, max_cost);
}
}  // namespace amr