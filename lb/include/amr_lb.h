#pragma once

#include <mpi.h>

#include <string>
#include <vector>

namespace amr {
namespace lb {
// PlacementArgs: minimal placement inputs and outputs
struct PlacementArgs {
  std::string policy_name;              // preconfigured policy name
  std::vector<double> const& costlist;  // size assumed to be nblocks
  std::vector<int>& ranklist;           // will be resized to nblocks
  int nranks;                           // number of policy ranks
};

// PlacementArgsMpi: PlacementArgs + MPI info for parallel placement
// (MPI  ranks here is separate from stdargs ranks, to decouple input and exec)
// (But this may not be implemented all the way)
struct PlacementArgsMpi {
  PlacementArgs stdargs;  // std args
  int my_rank;            // MPI rank
  int nranks;             // number of MPI ranks
  MPI_Comm comm;          // MPI communicator
};

//
// LoadBalance: top-level interface for placement
//
class LoadBalance {
 public:
  //
  // AssignBlocks: assign blocks to ranks using the given policy
  // Returns 0 on success, 1 on failure
  //
  static int AssignBlocks(PlacementArgs args);

  //
  // AssignBlocksMpi: assign blocks to ranks using the given policy in parallel
  // Returns 0 on success, 1 on failure
  //
  static int AssignBlocksMpi(PlacementArgsMpi args);
};
}  // namespace lb
}  // namespace amr