#pragma once

#include <mpi.h>
#include <vector>

namespace amr {
class HybridAssignmentCppFirst {
 public:
  HybridAssignmentCppFirst(int lpt_ranks, int alt_solncnt_max)
      : lpt_rank_count_(lpt_ranks), alt_max_(alt_solncnt_max) {}

  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks);

  int AssignBlocksV2(std::vector<double> const& costlist,
                     std::vector<int>& ranklist, int nranks, MPI_Comm comm);

  //
  // Get ranks to run LPT on. This is run after the initial CDP
  // assignment. Ranks are sorted by cost, and:
  // k highest-loaded ranks are selected, and then
  // a number of lowest loaded ranks are selected to ensure
  // that the average of selected ranks is close to average
  //
  // k === lpt_rank_count_
  //
  std::vector<int> GetLPTRanks(std::vector<double> const& costlist,
                               std::vector<int> const& ranklist) const;

  //
  // Get ranks to run LPT on. This is run after the initial CDP placement.
  // Unlike V1, V2 will always only select k total ranks.
  // It will automatically divide k into most and least loaded ranks
  // to the extent possible
  //
  // k === lpt_rank_count_
  //
  std::vector<int> GetLPTRanksV2(std::vector<double> const& costlist,
                                 std::vector<int> const& ranklist) const;

  //
  // Get ranks to run LPT on. This is run after the initial CDP placement.
  // Unlike V1 and V2, V3 will select a maximum of k ranks.
  // V3 uses a diff threshold (set to 2%) to determine whether to LPT
  // a rank or not. Thus, even for high values of k, it may not select
  // ranks if they are relatively balanced
  //
  static std::vector<int> GetLPTRanksV3(std::vector<double> const& costlist,
                                        std::vector<int> const& ranklist,
                                        std::vector<double> const& rank_costs,
                                        const int nmax);

  static std::vector<int> GetBlocksForRanks(
      std::vector<int> const& ranklist, std::vector<int> const& selected_ranks);

  constexpr static const char* kCDPPolicyStr = "cdpc512";
  constexpr static const char* kParCDPPolicyStr = "cdpc512par16";
  constexpr static const char* kLPTPolicyStr = "lpt";
  constexpr static double kLPTv3DiffThreshold = 0.02;

 private:
  const int lpt_rank_count_;
  const int alt_max_;

  int nblocks_;
  int nranks_;
  std::vector<double> rank_costs_;
};
}  // namespace amr
