#pragma once

#include <Profile/TauPluginTypes.h>
#include <TAU.h>

namespace tau {

enum class MsgType { kBlockAssignment, kTargetCost, kTsEnd, kEventTime };

struct TriggerMsg {
  MsgType type;
  void *data;
};

struct MsgBlockAssignment {
  std::vector<double> const *costlist;
  std::vector<int> const *ranklist;
};

struct MsgEventTime {
  int block_id;
  int event_opcode;
  int time_us;
};
} // namespace tau
