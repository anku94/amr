//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <iostream>
#include <sstream>
#include <math.h>
#include <numeric>

#include "common.h"

namespace amr {
class Policies {
 public:
  static void AssignBlocks(std::vector<double> const &costlist,
                           std::vector<int> &ranklist, int nranks) {
    ranklist.resize(costlist.size());

    // AssignBlocksRoundRobin(costlist, ranklist, nranks);
    // AssignBlocksSkewed(costlist, ranklist, nranks);
    AssignBlocksContiguous(costlist, ranklist, nranks);
  }

  static void AssignBlocksRoundRobin(std::vector<double> const &costlist,
                                     std::vector<int> &ranklist, int nranks) {
    logf(LOG_DBUG, "[AmrHacks] Assignment Policy: RoundRobin");

    for (int block_id = 0; block_id < costlist.size(); block_id++) {
      int block_rank = block_id % nranks;
      logf(LOG_DBG2, "Block %d: Rank %d", block_id, block_rank);

      ranklist[block_id] = block_rank;
    }

    return;
  }

  static void AssignBlocksSkewed(std::vector<double> const &costlist,
                                 std::vector<int> &ranklist, int nranks) {
    logf(LOG_DBUG, "[AmrHacks] Assignment Policy: Skewed");

    int nblocks = costlist.size();

    float avg_alloc = nblocks * 1.0f / nranks;
    int rank0_alloc = ceilf(avg_alloc);

    while ((nblocks - rank0_alloc) % (nranks - 1)) {
      rank0_alloc++;
    }

    if (rank0_alloc >= nblocks) {
      std::stringstream msg;
      msg << "### FATAL ERROR rank0_alloc >= nblocks "
          << "(" << rank0_alloc << ", " << nblocks << ")" << std::endl;
      ABORT(msg.str().c_str());
    }

    for (int bid = 0; bid < nblocks; bid++) {
      if (bid <= rank0_alloc) {
        ranklist[bid] = 0;
      } else {
        int rem_alloc = (nblocks - rank0_alloc) / (nranks - 1);
        int bid_adj = bid - rank0_alloc;
        ranklist[bid] = 1 + bid_adj / rem_alloc;
      }
    }

    return;
  }

  static void AssignBlocksContiguous(std::vector<double> const &costlist,
                                     std::vector<int> &ranklist, int nranks) {
    logf(LOG_DBUG, "[AmrHacks] Assignment Policy: Contiguous");

    double const total_cost = std::accumulate(costlist.begin(), costlist.end(), 0.0);

    int rank = nranks - 1;
    double target_cost = total_cost / nranks;
    double my_cost = 0.0;
    double remaining_cost = total_cost;
    // create rank list from the end: the master MPI rank should have less load
    for (int block_id = costlist.size() - 1; block_id >= 0; block_id--) {
      if (target_cost == 0.0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
            << "There is at least one process which has no MeshBlock" << std::endl
            << "Decrease the number of processes or use smaller MeshBlocks." << std::endl;
        ABORT(msg.str().c_str());
      }
      my_cost += costlist[block_id];
      ranklist[block_id] = rank;
      if (my_cost >= target_cost && rank > 0) {
        rank--;
        remaining_cost -= my_cost;
        my_cost = 0.0;
        target_cost = remaining_cost / (rank + 1);
      }
    }
  }
};
} // namespace parthenon
