//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include "common.h"
#include "policy_sim.h"

#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>

namespace amr {
enum class Policy;
class Policies {
 public:
  static void AssignBlocks(Policy policy, std::vector<double> const& costlist,
                           std::vector<int>& ranklist, int nranks);

 private:
  static void AssignBlocksRoundRobin(std::vector<double> const& costlist,
                                     std::vector<int>& ranklist, int nranks);

  static void AssignBlocksSkewed(std::vector<double> const& costlist,
                                 std::vector<int>& ranklist, int nranks);

  static void AssignBlocksContiguous(std::vector<double> const& costlist,
                                     std::vector<int>& ranklist, int nranks);
};
}  // namespace amr
