#include <mpi.h>

#include <vector>

#include "common.h"
#include "lb_policies.h"

namespace amr {
// fwd decl
class LBChunkwiseTest;

struct WorkloadChunk {
  int block_first;
  int block_last;
  int rank_first;
  int rank_last;
  double cost;

  WorkloadChunk()
      : block_first(0), block_last(0), rank_first(0), rank_last(0), cost(0) {}

  std::string ToString() const {
    char buf[2048];
    snprintf(buf, sizeof(buf), "B[%d, %d), R[%d, %d): C%.2lf", block_first,
             block_last, rank_first, rank_last, cost);
    return std::string(buf);
  }

  int NumBlocks() const { return block_last - block_first; }

  int NumRanks() const { return rank_last - rank_first; }
};

#define EPSILON 1e-12

class LBChunkwise {
 public:
  static int AssignBlocks(std::vector<double> const& costlist,
                          std::vector<int>& ranklist, int nranks, int nchunks) {
    auto chunks = ComputeChunks(costlist, nranks, nchunks);
    ValidateChunks(chunks, costlist.size(), nranks, nchunks);

    logv(__LOG_ARGS__, LOG_DBUG, "Computed %d chunks", chunks.size());

    int chunk_idx = 0;
    for (auto const& chunk : chunks) {
      logv(__LOG_ARGS__, LOG_DBG2, "Chunk %d: %s", chunk_idx++,
           chunk.ToString().c_str());

      std::vector<double> const chunk_costlist =
          std::vector<double>(costlist.begin() + chunk.block_first,
                              costlist.begin() + chunk.block_last);

      std::vector<int> chunk_ranklist(chunk.NumBlocks(), -1);
      int rv = LoadBalancePolicies::AssignBlocksContigImproved(
          chunk_costlist, chunk_ranklist, chunk.NumRanks());

      if (rv != 0) {
        logv(__LOG_ARGS__, LOG_WARN,
             "Failed to assign blocks to chunk %s, rv: %d",
             chunk.ToString().c_str(), rv);

        return rv;
      }

      for (int i = 0; i < chunk.NumBlocks(); i++) {
        ranklist[chunk.block_first + i] = chunk.rank_first + chunk_ranklist[i];
      }
    }

    return 0;
  }

  static int AssignBlocksParallel(std::vector<double> const& costlist,
                                  std::vector<int>& ranklist, int nranks,
                                  MPI_Comm comm, int mympirank, int nmpiranks,
                                  int nchunks) {
    if (nchunks > nranks or nchunks > nmpiranks) {
      logv(__LOG_ARGS__, LOG_ERRO, "nchunks > nranks");
      ABORT("nchunks > nranks");
      return -1;
    }

    auto chunks = ComputeChunks(costlist, nranks, nchunks);
    ValidateChunks(chunks, costlist.size(), nranks, nchunks);

    std::vector<int> chunk_ranklist;

    if (mympirank < nchunks) {
      auto const& chunk = chunks[mympirank];
      logv(__LOG_ARGS__, LOG_DBUG, "Rank %d: Executing chunk %s", mympirank,
           chunk.ToString().c_str());

      std::vector<double> const chunk_costlist =
          std::vector<double>(costlist.begin() + chunk.block_first,
                              costlist.begin() + chunk.block_last);

      chunk_ranklist.resize(chunk.NumBlocks(), -1);
      int rv = LoadBalancePolicies::AssignBlocksContigImproved(
          chunk_costlist, chunk_ranklist, chunk.NumRanks());

      if (rv != 0) {
        logv(__LOG_ARGS__, LOG_ERRO,
             "Failed to assign blocks to chunk %s, rv: %d",
             chunk.ToString().c_str(), rv);

        return rv;
      }

      // offset chunk block-rank mappings by chunk rank_first
      for (int i = 0; i < chunk.NumBlocks(); i++) {
        chunk_ranklist[i] = chunk.rank_first + chunk_ranklist[i];
      }
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Rank %d: Gathering results", mympirank);

    // Gather the results
    // First, prepare recvcnts and displs
    std::vector<int> recvcnts(nmpiranks, 0);
    for (int rank = 0; rank < nmpiranks; rank++) {
      recvcnts[rank] = (rank < nchunks) ? chunks[rank].NumBlocks() : 0;
    }

    std::vector<int> displs(nmpiranks, 0);
    for (int i = 1; i < nmpiranks; i++) {
      displs[i] = displs[i - 1] + recvcnts[i - 1];
    }

    int sendcnt = (mympirank < nchunks) ? chunks[mympirank].NumBlocks() : 0;

    int rv =
        MPI_Allgatherv(chunk_ranklist.data(), sendcnt, MPI_INT, ranklist.data(),
                       recvcnts.data(), displs.data(), MPI_INT, comm);

    if (rv != MPI_SUCCESS) {
      logv(__LOG_ARGS__, LOG_WARN, "MPI_Allgatherv failed, rv: %d", rv);
      return rv;
    }

    return 0;
  }

 private:
  static std::string SerializeVector(std::vector<int> const& v) {
    char buf[65536];
    char* bufptr = buf;
    int bytes_rem = sizeof(buf);

    for (int i = 0; i < v.size(); i++) {
      int r = snprintf(bufptr, bytes_rem, "%5d ", v[i]);
      bufptr += r;
      bytes_rem -= r;
      if (i % 16 == 15) {
        r = snprintf(bufptr, bytes_rem, "\n");
        bufptr += r;
        bytes_rem -= r;
      }
    }
    return std::string(buf);
  }
  static std::vector<WorkloadChunk> ComputeChunks(
      std::vector<double> const& costlist, int nranks, int nchunks) {
    if (nranks % nchunks != 0) {
      ABORT("nranks must be divisible by nchunks");
      return {};
    }

    logv(__LOG_ARGS__, LOG_DBUG, "nranks: %d, nchunks: %d", nranks, nchunks);
    int nblocks = costlist.size();
    int nranks_pc = nranks / nchunks;

    std::vector<WorkloadChunk> chunks(nchunks);

    // Initialize vector of chunks with known ranks
    for (int cidx = 0; cidx < nchunks; cidx++) {
      chunks[cidx].rank_first = cidx * nranks_pc;
      chunks[cidx].rank_last = (cidx + 1) * nranks_pc;
    }

    // Initialize with block ranges
    chunks[0].block_first = 0;
    chunks[nchunks - 1].block_last = nblocks;

    double cost_total = std::accumulate(costlist.begin(), costlist.end(), 0.0);
    double cost_pc = cost_total / nchunks;

    logv(__LOG_ARGS__, LOG_DBUG, "Cost total: %.2lf, per-chunk: %.2lf",
         cost_total, cost_pc);

    int cur_cidx = 0; // current chunk index
    double cur_cost = 0; // current cost

    // target_cost is cost_pc, but is calculated from cost_remaining
    // to avoid floating point errors
    double cost_rem = cost_total;
    double target_cost = cost_rem / nchunks;

    for (int bidx = 0; bidx < nblocks; bidx++) {
      // Add current block to the chunk
      cur_cost += costlist[bidx];

      int nblocks_rem = nblocks - bidx - 1;
      int nchunks_rem = nchunks - cur_cidx - 1;
      int cur_count = bidx - chunks[cur_cidx].block_first + 1;

      // Breaking conditions:
      // Cond1. Current chunk has enough cost and 1 block per rank
      // Cond2. Need to break to ensure at least 1 block for remaining ranks
      bool cond1 =
          (cur_cost >= target_cost - EPSILON and cur_count >= nranks_pc);
      bool cond2 = (nblocks_rem <= nchunks_rem * nranks_pc);

      logv(__LOG_ARGS__, LOG_DBG3,
           "[%4d] Cond1 (%s): %.2lf >= %.2lf and %d >= %d", bidx,
           cond1 ? "true" : "false", cur_cost, target_cost, cur_count,
           nranks_pc);

      logv(__LOG_ARGS__, LOG_DBG3, "[%4d] Cond2 (%s): %d <= %d * %d", bidx,
           cond2 ? "true" : "false", nblocks_rem, nchunks_rem, nranks_pc);

      if (cond1 or cond2) {
        // Update boundary between current and next chunk on both chunks
        chunks[cur_cidx].block_last = bidx + 1;
        if (cur_cidx < nchunks - 1) {
          chunks[cur_cidx + 1].block_first = bidx + 1;
        }

        chunks[cur_cidx].cost = cur_cost;

        logv(__LOG_ARGS__, LOG_DBUG, "Chunk %d: %s", cur_cidx,
             chunks[cur_cidx].ToString().c_str());

        // move on to next chunk, update target cost
        cur_cidx++;
        cost_rem -= cur_cost;
        target_cost = cost_rem / nchunks_rem;
        cur_cost = 0;
      }
    }

    for (int cidx = 0; cidx < nchunks; cidx++) {
      logv(__LOG_ARGS__, LOG_DBUG, "Chunk %d: %s", cidx,
           chunks[cidx].ToString().c_str());
    }

    return chunks;
  }

#define ASSERT(cond)                   \
  if (!(cond)) {                       \
    ABORT("Assertion failed: " #cond); \
  }

  static void ValidateChunks(std::vector<WorkloadChunk> const& chunks,
                             int nblocks, int nranks, int nchunks) {
    if (chunks.size() != nchunks) {
      logv(__LOG_ARGS__, LOG_WARN, "Expected %d chunks, got %d", nchunks,
           chunks.size());
      ABORT("Chunk count mismatch");
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Chunk range: %d %d", chunks[0].block_first,
         chunks[nchunks - 1].block_last);

    ASSERT(chunks[0].block_first == 0);
    ASSERT(chunks[nchunks - 1].block_last == nblocks);

    int nranks_pc = nranks / nchunks;

    for (int cidx = 0; cidx < nchunks; cidx++) {
      auto const& chunk = chunks[cidx];

      ASSERT(chunk.block_first >= 0 && chunk.block_first < nblocks);
      ASSERT(chunk.block_last > chunk.block_first &&
             chunk.block_last <= nblocks);
      ASSERT(chunk.rank_first == cidx * nranks_pc);
      ASSERT(chunk.rank_last == (cidx + 1) * nranks_pc);

      ASSERT(chunk.NumBlocks() >= nranks_pc);

      if (cidx < nchunks - 1) {
        ASSERT(chunks[cidx].block_last == chunks[cidx + 1].block_first);
      }
    }
  }

#undef ASSERT

  friend class LBChunkwiseTest;
};
}  // namespace amr
