#pragma once

#include <mpi.h>

#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "lb_policies.h"

namespace amr {
struct RunType {
  int nranks;
  int nblocks;
  std::string policy;

  std::string ToString() const {
    std::stringstream ss;
    ss << std::string(60, '-') << "\n";
    ss << "[BenchmarkRun] policy: " << policy << ", n_ranks: " << nranks
       << ", n_blocks: " << nblocks;
    return ss.str();
  }

  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks) {
    int rv = LoadBalancePolicies::AssignBlocks(policy.c_str(), costlist,
                                               ranklist, nranks);
    return rv;
  }

  int AssignBlocksParallel(std::vector<double> const& costlist,
                           std::vector<int>& ranklist, int nranks,
                           MPI_Comm comm) {
    int rv = LoadBalancePolicies::AssignBlocksParallel(
        policy.c_str(), costlist, ranklist, nranks, comm);
    return rv;
  }

  static void VerifyAssignment(std::vector<int> const& ranklist, int nranks) {
    std::vector<int> counts(nranks, 0);
    for (auto const& rank : ranklist) {
      if (rank == -1) {
        ABORT("All blocks must be assigned");
      }

      counts[rank]++;
    }

    //   for (auto const& count : counts) {
    //     if (count == 0) {
    //       ABORT("All ranks must have some assignment");
    //     }
    //   }
  }
};
}  // namespace amr
