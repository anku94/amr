#pragma once

#include <vector>

namespace amr {
class HybridAssignmentCppFirst {
 public:
  HybridAssignmentCppFirst(int lpt_ranks) : lpt_rank_count_(lpt_ranks) {}

  int AssignBlocks(std::vector<double> const& costlist,
                   std::vector<int>& ranklist, int nranks);

  int AssignBlocksV2(std::vector<double> const& costlist,
                     std::vector<int>& ranklist, int nranks);

  std::vector<int> GetLPTRanks(std::vector<double> const& costlist,
                               std::vector<int> const& ranklist) const;

  std::vector<int> GetLPTRanksV2(std::vector<double> const& costlist,
                                 std::vector<int> const& ranklist) const;

  static std::vector<int> GetLPTRanksV3(std::vector<double> const& costlist,
                                        std::vector<int> const& ranklist,
                                        std::vector<double> const& rank_costs,
                                        const int nmax);

  static std::vector<int> GetBlocksForRanks(
      std::vector<int> const& ranklist, std::vector<int> const& selected_ranks);

 private:
  const int lpt_rank_count_;

  int nblocks_;
  int nranks_;
  std::vector<double> rank_costs_;
};
}  // namespace amr
