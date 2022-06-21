#pragma once

#include <TAU.h>
#include <Profile/TauPluginTypes.h>

namespace tau {

enum class MsgType { kBlockAssignment, kTargetCost };

struct TriggerMsg {
  MsgType type;
  void* data;
};

struct MsgBlockAssignment {
  std::vector<double> const *costlist;
  std::vector<int> const* ranklist;
};
} // namespace tau

