#pragma once

#include <Profile/TauPluginTypes.h>
#include <TAU.h>

namespace tau {

enum class MsgType {
  kBlockAssignment,
  kTargetCost,
  kTsEnd,
  kBlockEvent,
  kCommChannel,
  kMsgSend
};

struct TriggerMsg {
  MsgType type;
  void* data;
};

struct MsgBlockAssignment {
  std::vector<double> const* costlist;
  std::vector<int> const* ranklist;
};

struct MsgBlockEvent {
  int block_id;
  int opcode;
  int data;
};

struct MsgCommChannel {
  void* ptr;
  int block_id;
  int block_rank;
  int nbr_id;
  int nbr_rank;
  int tag;
  char is_flux;
};

struct MsgSend {
  void* ptr;
  int buf_sz;
  int recv_rank;
  int tag;
  uint64_t timestamp;
};
}  // namespace tau
