#include "lb_chunkwise.h"

#include "common.h"
#include "lb_policies.h"
#include "policy_wopts.h"

namespace amr {
int LoadBalancePolicies::AssignBlocksCDPChunked(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    PolicyOptsChunked const& opts) {
  int chunksz = opts.chunk_size;

  if (nranks % chunksz != 0) {
    logv(__LOG_ARGS__, LOG_WARN,
         "Number of ranks %d is not a multiple of chunk size %d", nranks,
         chunksz);
    return -1;
  }

  int nchunks = (nranks > chunksz) ? nranks / chunksz : 1;
  int rv = LBChunkwise::AssignBlocks(costlist, ranklist, nranks, nchunks);

  if (rv) {
    logv(__LOG_ARGS__, LOG_WARN, "Failed to assign blocks to chunks, rv: %d",
         rv);
  }

  return rv;
}

int LoadBalancePolicies::AssignBlocksParallelCDPChunked(
    std::vector<double> const& costlist, std::vector<int>& ranklist, int nranks,
    PolicyOptsChunked const& opts, MPI_Comm comm, int mympirank,
    int nmpiranks) {
  int chunksz = opts.chunk_size;

  // we only care about this if we have more ranks than one chunk
  if (nranks % chunksz != 0 and nranks > chunksz) {
    logv(__LOG_ARGS__, LOG_WARN,
         "Number of ranks %d is not a multiple of chunk size %d", nranks,
         chunksz);
    return -1;
  }

  int parallelism = std::min(opts.parallelism, nmpiranks);
  int nchunks_max = nranks / chunksz;

  int nchunks = parallelism;
  if (nchunks_max > 0) {
    nchunks = std::min(nchunks_max, parallelism);
  }

  // int nchunks = (nranks > chunksz) ? nranks / chunksz : 1;
  int rv = LBChunkwise::AssignBlocksParallel(costlist, ranklist, nranks, comm,
                                             mympirank, nmpiranks, nchunks);

  if (rv) {
    logv(__LOG_ARGS__, LOG_WARN, "Failed to assign blocks to chunks, rv: %d",
         rv);
  }

  return rv;
}
}  // namespace amr
