#pragma once

#include <Profile/TauPluginTypes.h>
#include <TAU.h>

namespace tau {

enum class MsgType { kBlockAssignment, kTargetCost, kTsEnd, kBlockEvent };

struct TriggerMsg {
  MsgType type;
  void *data;
};

struct MsgBlockAssignment {
  std::vector<double> const *costlist;
  std::vector<int> const *ranklist;
};

struct MsgBlockEvent {
  int block_id;
  int opcode;
  int data;
};
} // namespace tau
