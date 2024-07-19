// define a gtest skeleton

#include <gtest/gtest.h>

#include "lb/lb_chunkwise.h"

namespace amr {
class LBChunkwiseTest : public ::testing::Test {
 protected:
  std::vector<WorkloadChunk> ComputeChunks(std::vector<double> const& costlist,
                                           int nranks, int nchunks) {
    return LBChunkwise::ComputeChunks(costlist, nranks, nchunks);
  }

  void ValidateChunks(std::vector<WorkloadChunk> const& chunks, int nchunks,
                      int nblocks, int nranks) {
    int nchunks_v = chunks.size();
    EXPECT_EQ(nchunks_v, nchunks);

    EXPECT_EQ(chunks[0].block_first, 0);
    EXPECT_EQ(chunks[nchunks - 1].block_last, nblocks);

    int nranks_pc = nranks / nchunks;

    for (int cidx = 0; cidx < nchunks; cidx++) {
      auto const& chunk = chunks[cidx];
      EXPECT_EQ(chunk.rank_first, cidx * nranks_pc);
      EXPECT_EQ(chunk.rank_last, (cidx + 1) * nranks_pc);
      EXPECT_GE(chunk.NumBlocks(), nranks_pc);

      if (cidx < nchunks - 1) {
        EXPECT_EQ(chunk.block_last, chunks[cidx + 1].block_first);
      }
    }
  }
};

TEST_F(LBChunkwiseTest, ValidateChunking) {
  int nblocks = 6;
  int nranks = 4;
  int nchunks = 2;

  std::vector<double> costs = {2.0, 3.0, 2.0, 3.0, 2.0, 3.0};

  auto chunks = ComputeChunks(costs, nranks, nchunks);
  ValidateChunks(chunks, nchunks, nblocks, nranks);
}

TEST_F(LBChunkwiseTest, ValidateChunking2) {
  int nblocks = 2000;
  int nranks = 512;
  int nchunks = 4;
  std::vector<double> costs(nblocks, 1.0);

  auto chunks = ComputeChunks(costs, nranks, nchunks);
  ValidateChunks(chunks, nchunks, nblocks, nranks);

  nblocks=1999;
  costs.resize(nblocks, 1.0);
  chunks = ComputeChunks(costs, nranks, nchunks);
  ValidateChunks(chunks, nchunks, nblocks, nranks);

  costs = {2, 2, 2, 3, 3, 2, 2, 2, 3, 3};
  nranks = 4;
  nchunks = 4;
  nblocks = costs.size();
  chunks = ComputeChunks(costs, nranks, nchunks);
  ValidateChunks(chunks, nchunks, nblocks, nranks);
}

TEST_F(LBChunkwiseTest, ValidateChunking3) {
  int nblocks = 8;
  int nranks = 6;
  int nchunks = 3;
  std::vector<double> costs(nblocks, 1e-6);

  auto chunks = ComputeChunks(costs, nranks, nchunks);
  ValidateChunks(chunks, nchunks, nblocks, nranks);

  for (const auto& chunk: chunks) {
    EXPECT_GE(chunk.NumBlocks(), 2);
  }
}

TEST_F(LBChunkwiseTest, ValidateChunking4) {
  int nblocks = 4195;
  int nranks = 4096;
  int nchunks = 8;
  std::vector<double> costs(nblocks, 1e-6);
  costs[4095] = 1;
  costs[4083] = 3;
  costs[2] = 100;

  auto chunks = ComputeChunks(costs, nranks, nchunks);
  ValidateChunks(chunks, nchunks, nblocks, nranks);

  for (const auto& chunk: chunks) {
    EXPECT_GE(chunk.NumBlocks(), 512);
  }
}

TEST_F(LBChunkwiseTest, ValidateChunking5) {
  int nblocks = 9150;
  int nranks = 4096;
  int nchunks = 4;
  std::vector<double> costs(nblocks, 1e-6);
  costs[4095] = 1;
  costs[4083] = 3;
  costs[2] = 100;

  auto chunks = ComputeChunks(costs, nranks, nchunks);

  for (const auto& chunk: chunks) {
    EXPECT_GE(chunk.NumBlocks(), 1024);
  }

  ValidateChunks(chunks, nchunks, nblocks, nranks);
}
}  // namespace amr
