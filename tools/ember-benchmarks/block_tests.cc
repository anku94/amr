#include <algorithm>
#include <gtest/gtest.h>
#include "block_common.h"

class BlockTests : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up the test case
  }

  void TearDown() override {
    // Tear down the test case
  }
};

// Write a HelloWorld test
TEST_F(BlockTests, HelloWorld) {
  EXPECT_EQ(1, 1);
}

TEST_F(BlockTests, EdgeNeighborTest) {
  const Triplet bounds(8, 8, 8);
  const Triplet my = PositionUtils::GetPosition(200, bounds);

  NeighborRankGenerator nrg(my, bounds);
  std::vector<int> neighbors = nrg.GetEdgeNeighbors();
  std::vector<int> neighbors_expected = {
    -1, -1, -1, -1, 128, 137, 144, 193, 209, 256, 265, 272
  };

  std::sort(neighbors.begin(), neighbors.end());

  for (auto n: neighbors) {
    LOG(INFO) << "Neighbor: " << n;
  }

  EXPECT_EQ(neighbors, neighbors_expected);

  return;
}
