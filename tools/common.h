//
// Created by Ankush J on 4/11/22.
//

#pragma once

#include <stdio.h>

enum class Status {
  OK,
  MPIError,
  Error
};

namespace Globals {
extern int my_rank, nranks;
};
enum class NeighborTopology { Ring,
                              AllToAll };

struct DriverOpts {
  NeighborTopology topology;
  size_t blocks_per_rank;
  size_t msgs_per_block;
  size_t size_per_msg;
};
