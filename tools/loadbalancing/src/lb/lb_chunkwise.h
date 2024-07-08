#include <vector>

#include "common.h"
#include "lb_policies.h"

namespace amr {
// fwd decl
class LBChunkwiseTest;

struct WorkloadChunk {
  int block_first;
  int nblocks;
  int rank_first;
  int nranks;

  std::string ToString() const {
    char buf[2048];
    snprintf(buf, sizeof(buf), "B[%d, %d], R[%d, %d]", block_first,
             block_first + nblocks - 1, rank_first, rank_first + nranks - 1);
    return std::string(buf);
  }
};

#define EPSILON 1e-6

class LBChunkwise {
 public:
  static int AssignBlocks(std::vector<double> const& costlist,
                    std::vector<int>& ranklist, int nranks, int nchunks) {
    auto chunks = ComputeChunks(costlist, nranks, nchunks);

    for (auto const& chunk : chunks) {
      std::vector<double> const chunk_costlist = std::vector<double>(
          costlist.begin() + chunk.block_first,
          costlist.begin() + chunk.block_first + chunk.nblocks);

      std::vector<int> chunk_ranklist(chunk.nblocks, -1);
      int rv = LoadBalancePolicies::AssignBlocksContigImproved(
          chunk_costlist, chunk_ranklist, chunk.nranks);

      if (rv != 0) {
        logv(__LOG_ARGS__, LOG_WARN,
             "Failed to assign blocks to chunk %s, rv: %d",
             chunk.ToString().c_str(), rv);

        return rv;
      }

      for (int i = 0; i < chunk.nblocks; i++) {
        ranklist[chunk.block_first + i] = chunk.rank_first + chunk_ranklist[i];
      }
    }

    return 0;
  }

 private:
  static std::vector<WorkloadChunk> ComputeChunks(
      std::vector<double> const& costlist, int nranks, int nchunks) {
    // costlist[]: size is nblocks
    // first, create a cumulative array for costlist
    // then, assign nchunks worth of work and nranks/nchunks ranks to each chunk

    if (nranks % nchunks != 0) {
      ABORT("nranks must be divisible by nchunks");
      return {};
    }

    logv(__LOG_ARGS__, LOG_DBUG, "nranks: %d, nchunks: %d", nranks, nchunks);

    std::vector<WorkloadChunk> chunks;
    int nblocks = costlist.size();
    int nranks_pc = nranks / nchunks;

    std::vector<double> cum_costlist(nblocks);
    cum_costlist[0] = costlist[0];
    for (int i = 1; i < nblocks; i++) {
      cum_costlist[i] = cum_costlist[i - 1] + costlist[i];
    }

    double cost_total = cum_costlist[nblocks - 1];
    double cost_pc = cost_total / nchunks;

    logv(__LOG_ARGS__, LOG_DBUG, "Cost total: %.2lf, per-chunk: %.2lf",
         cost_total, cost_pc);

    double cost_rem = cost_total;
    double cost_to_target = cost_pc;
    double cost_cur = 0;
    int nblocks_beg = 0;
    int nranks_beg = 0;

    for (int bidx = 0; bidx < nblocks; bidx++) {
      cost_cur += costlist[bidx];

      if (cost_cur >= cost_to_target - EPSILON) {
        chunks.push_back({
            .block_first = nblocks_beg,
            .nblocks = bidx - nblocks_beg + 1,
            .rank_first = nranks_beg,
            .nranks = nranks_pc,
        });

        logv(__LOG_ARGS__, LOG_DBUG, "Chunk %d: %s, cost: %.2lf", chunks.size(),
             chunks.back().ToString().c_str(), cost_cur);

        nblocks_beg = bidx + 1;
        nranks_beg += nranks_pc;
        cost_rem -= cost_cur;
        cost_cur = 0;
        cost_to_target = cost_rem / (nchunks - chunks.size());
      }
    }

    if (chunks.size() != nchunks) {
      logv(__LOG_ARGS__, LOG_WARN,
           "Something went wrong with chunk assignment."
           "(Expected %d chunks, got %d)",
           nchunks, chunks.size());
      return {};
    }

    return chunks;
  }

  friend class LBChunkwiseTest;
};
}  // namespace amr
