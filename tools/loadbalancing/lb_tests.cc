//
// Created by Ankush J on 4/10/23.
//

#include "lb_policies.h"

#include <gtest/gtest.h>

namespace amr {
class LoadBalancingPoliciesTest : public ::testing::Test {
 protected:
  void AssignBlocksSPT(std::vector<double> const& costlist,
                       std::vector<int>& ranklist, int nranks) {
    LoadBalancePolicies::AssignBlocksSPT(costlist, ranklist, nranks);
  }

  void AssignBlocksLPT(std::vector<double> const& costlist,
                       std::vector<int>& ranklist, int nranks) {
    LoadBalancePolicies::AssignBlocksLPT(costlist, ranklist, nranks);
  }
};

TEST_F(LoadBalancingPoliciesTest, SPTTest1) {
  logf(LOG_INFO, "SPT Test 1");

  std::vector<double> costlist = { 1, 2, 3, 4 };
  std::vector<int> ranklist;
  ranklist.resize(costlist.size());

  int nranks = 2;
  AssignBlocksSPT(costlist, ranklist, nranks);

  std::string ranklist_str = SerializeVector(ranklist);
  logf(LOG_INFO, "Assignment: %s", ranklist_str.c_str());

  ASSERT_EQ(ranklist[0], ranklist[2]);
  ASSERT_EQ(ranklist[1], ranklist[3]);
}

TEST_F(LoadBalancingPoliciesTest, LPTTest1) {
  logf(LOG_INFO, "LPT Test 1");

  std::vector<double> costlist = { 1, 2, 3, 4 };
  std::vector<int> ranklist;
  ranklist.resize(costlist.size());

  int nranks = 2;
  AssignBlocksLPT(costlist, ranklist, nranks);

  std::string ranklist_str = SerializeVector(ranklist);
  logf(LOG_INFO, "Assignment: %s", ranklist_str.c_str());

  ASSERT_EQ(ranklist[0], ranklist[3]);
  ASSERT_EQ(ranklist[1], ranklist[2]);
}
}  // namespace amr
