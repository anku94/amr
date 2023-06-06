//
// Created by Ankush J on 5/8/23.
//

#include "common.h"
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
    return LoadBalancePolicies::AssignBlocksLPT(costlist, ranklist,
                                                       nranks);
  }

  static int AssignBlocksContigImproved(std::vector<double> const& costlist,
                             std::vector<int>& ranklist, int nranks) {
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
  std::vector<double> costlist = { 1, 2, 3, 2, 1};
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
}  // namespace amr