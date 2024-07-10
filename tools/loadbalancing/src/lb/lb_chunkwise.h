#include <mpi.h>

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
    ValidateChunks(chunks);

    logv(__LOG_ARGS__, LOG_DBG2, "Computed %d chunks", chunks.size());

    int chunk_idx = 0;

    for (auto const& chunk : chunks) {
      logv(__LOG_ARGS__, LOG_DBG2, "Chunk %d: %s", chunk_idx++,
           chunk.ToString().c_str());

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
    ValidateChunks(chunks);

    std::vector<int> chunk_ranklist;

    if (mympirank < nchunks) {
      auto const& chunk = chunks[mympirank];
      logv(__LOG_ARGS__, LOG_DBUG, "Rank %d: Executing chunk %s", mympirank,
           chunk.ToString().c_str());

      std::vector<double> const chunk_costlist = std::vector<double>(
          costlist.begin() + chunk.block_first,
          costlist.begin() + chunk.block_first + chunk.nblocks);

      chunk_ranklist.resize(chunk.nblocks, -1);
      int rv = LoadBalancePolicies::AssignBlocksContigImproved(
          chunk_costlist, chunk_ranklist, chunk.nranks);

      if (rv != 0) {
        logv(__LOG_ARGS__, LOG_ERRO,
             "Failed to assign blocks to chunk %s, rv: %d",
             chunk.ToString().c_str(), rv);

        return rv;
      }

      for (int i = 0; i < chunk.nblocks; i++) {
        chunk_ranklist[i] = chunk.rank_first + chunk_ranklist[i];
      }
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Rank %d: Gathering results", mympirank);

    // Gather the results
    // First, prepare recvcnts and displs
    std::vector<int> recvcnts(nmpiranks, 0);
    for (int rank = 0; rank < nmpiranks; rank++) {
      recvcnts[rank] = (rank < nchunks) ? chunks[rank].nblocks : 0;
    }

    std::vector<int> displs(nmpiranks, 0);
    for (int i = 1; i < nmpiranks; i++) {
      displs[i] = displs[i - 1] + recvcnts[i - 1];
    }

    int sendcnt = (mympirank < nchunks) ? chunks[mympirank].nblocks : 0;

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

    // nranks remaining for subsequent chunks
    int nranks_rem = nranks - nranks_pc;
    int nblocks_rem = nblocks;

    if (nblocks_rem < nranks_rem) {
      logv(__LOG_ARGS__, LOG_WARN, "nblocks_rem < nranks_rem!!!");
    }

    for (int bidx = 0; bidx < nblocks; bidx++) {
      cost_cur += costlist[bidx];

      // this block is definitely going into this chunk
      // so will def not be available for the next
      nblocks_rem -= 1;

      // chunk end conditions, enumerated explicitly
      bool chunk_has_target_cost = (cost_cur >= cost_to_target - EPSILON);
      int nblocks_chunk = bidx - nblocks_beg + 1;
      bool chunk_has_target_nblocks = (nblocks_chunk >= nranks_pc);
      bool chunk_satisfies_target =
          (chunk_has_target_cost and chunk_has_target_nblocks);
      bool low_on_blocks = (nblocks_rem == nranks_rem);

      // the second condition here ensures that
      // each chunk gets at least nranks_pc blocks
      if (chunk_satisfies_target or low_on_blocks) {
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
        nranks_rem -= nranks_pc;
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

  static void ValidateChunks(std::vector<WorkloadChunk> const& chunks) {
    for (const auto& chunk : chunks) {
      if (chunk.nblocks < chunk.nranks) {
        logv(__LOG_ARGS__, LOG_WARN, "Chunk %s has fewer blocks than ranks",
             chunk.ToString().c_str());
        // A rank with zero blocks is not allowed
        ABORT("Chunk has fewer blocks than ranks");
      }
    }
  }

  friend class LBChunkwiseTest;
};
}  // namespace amr
