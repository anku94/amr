#include "amr_tracer.h"

#include "tau_types.h"

namespace {

#define BUFSZ 65536

std::string JoinVec(std::vector<double> const& v) {
  char buf[BUFSZ];
  size_t ptr = 0;
  size_t rem = BUFSZ;

  for (double k : v) {
    int cur = snprintf(buf + ptr, rem, "%.2lf,", k);
    if (cur > rem) break;
    ptr += cur;
    rem -= cur;
  }

  return buf;
}

std::string JoinVec(std::vector<int> const& v) {
  char buf[BUFSZ];
  size_t ptr = 0;
  size_t rem = BUFSZ;

  for (int k : v) {
    int cur = snprintf(buf + ptr, rem, "%d,", k);
    if (cur > rem) break;
    ptr += cur;
    rem -= cur;
  }

  return buf;
}
}  // namespace

namespace tau {

void AMRTracer::ProcessTriggerMsg(void* data) {
  TriggerMsg* msg = (TriggerMsg*)data;
  switch (msg->type) {
    case MsgType::kBlockAssignment:
      ProcessTriggerMsgBlockAssignment(msg->data);
      break;
    case MsgType::kTargetCost:
      ProcessTriggerMsgTargetCost(msg->data);
      break;
    default:
      logf(LOG_ERRO, "Unknown trigger msg type!");
      break;
  }

  return;
}

void AMRTracer::ProcessTriggerMsgBlockAssignment(void* data) {
  MsgBlockAssignment* msg = (MsgBlockAssignment*)data;

  std::string clstr = JoinVec(*(msg->costlist));
  std::string rlstr = JoinVec(*(msg->ranklist));

  logf(LOG_DBG2, "[Rank %d: CL] %s\n", rank_, clstr.c_str());
  logf(LOG_DBG2, "[Rank %d: RL] %s\n", rank_, rlstr.c_str());

  statelog_->LogKV(timestep_, "CL", clstr.c_str());
  statelog_->LogKV(timestep_, "RL", rlstr.c_str());
}

void AMRTracer::processTriggerMsgTargetCost(void* data) {
  double target_cost = *(double*)data;
  std::string cost_str = std::to_string(target_cost);
  statelog_->LogKV(timestep_, "TC", cost_str.c_str());
}
}  // namespace tau
