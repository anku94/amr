#include <algorithm>
#include <gtest/gtest.h>
#include "block_common.h"

class BlockTests : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up the test case
    int nranks = 512;
    const char* map_file = "map.txt";

    GenMapFile(nranks, map_file);
    RANK_MAP = std::make_unique<RankMap>(nranks, "map.txt");
  }

  void TearDown() override {
    // Tear down the test case
    const char* map_file = "map.txt";
    CleanMapFile(map_file);
  }

  void GenMapFile(int nranks, const char* map_fname) {
    FILE* f = fopen(map_fname, "w+");
    LOG_IF(FATAL, f == nullptr) << "Failed to open map file";

    for (int i = 0; i < nranks; i++) {
      fprintf(f, "%d,%d\n", i, (i + 1) % nranks);
    }
    fclose(f);
  }

  void CleanMapFile(const char* map_fname) {
    unlink(map_fname);
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
  std::vector<int> neighbors_expected_log = {
    -1, -1, -1, -1, 128, 137, 144, 193, 209, 256, 265, 272
  };

  std::vector<int> neighbors_expected_phy;
  for (auto n: neighbors_expected_log) {
    neighbors_expected_phy.push_back(RANK_MAP->GetPhysicalRank(n));
  }

  std::sort(neighbors.begin(), neighbors.end());

  for (auto n: neighbors) {
    LOG(INFO) << "Neighbor: " << n;
  }

  EXPECT_EQ(neighbors, neighbors_expected_phy);

  return;
}

TEST_F(BlockTests, RankMapTest) {
  int nranks = 512;
  for (int i = 0; i < nranks; i++) {
    ASSERT_EQ(RANK_MAP->GetLogicalRank(i), (i + 1) % nranks);
    ASSERT_EQ(RANK_MAP->GetPhysicalRank(i), (nranks + i - 1) % nranks);
  }
}
