#pragma once

#include "../tools/common/common.h"
#include "amr_outputs.h"
#include "amr_util.h"

#include <dirent.h>
#include <inttypes.h>
#include <memory>
#include <mpi/mpi.h>
#include <mutex>
#include <sys/types.h>

namespace {
  const char* GetLogDir() {
    const char* dir = getenv("TAU_AMR_LOGDIR");

    if (dir == nullptr) {
      ABORT("TAU_AMR_LOGDIR not set!");
    }

    DIR* dir_handle = opendir(dir);
    if (dir_handle) {
      closedir(dir_handle);
    } else {
      logf(LOG_ERRO, "Unable to verify TAU_AMR_LOGDIR: %s", strerror(errno));
      ABORT("Unable to verify TAU_AMR_LOGDIR");
    }

    return dir;
  }
}

namespace tau {

enum class AMRPhase { FluxExchange, LoadBalancing, BoundaryComm };

class AMRTracer {
 public:
  AMRTracer()
      : rank_(-1),
        size_(0),
        timestep_(0),
        ts_sim_(0),
        num_redistrib_(0),
        phase_(AMRPhase::LoadBalancing),
        redistribute_ongoing_(false) {
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    PMPI_Comm_size(MPI_COMM_WORLD, &size_);

    const char* dir = GetLogDir();

    msglog_ = std::make_unique<MsgLog>(dir, rank_);
    funclog_ = std::make_unique<FuncLog>(dir, rank_);
    statelog_ = std::make_unique<StateLog>(dir, rank_);
    proflog_ = std::make_unique<ProfLog>(dir, rank_);
  }

  int MyRank() const { return rank_; }

  void MarkBegin(const char* block_name, uint64_t ts) {
    funclog_->LogFunc(block_name, timestep_, ts, true);

    AmrFunc func = ParseBlock(block_name);
    switch (func) {
      case AmrFunc::RedistributeAndRefine:
        MarkRedistributeBegin();
        break;
      case AmrFunc::SendBoundBuf:
        MarkSendBoundBufBegin();
        break;
      case AmrFunc::RecvBoundBuf:
        MarkRecvBoundBufBegin();
        break;
      case AmrFunc::SendFluxCor:
        MarkSendFluxCorBegin();
        break;
      case AmrFunc::RecvFluxCor:
        MarkRecvFluxCorBegin();
        break;
      case AmrFunc::MakeOutputs:
        MarkMakeOutputsBegin();
        break;
      default:
        break;
    }
  }

  void MarkEnd(const char* block_name, uint64_t ts) {
    funclog_->LogFunc(block_name, timestep_, ts, false);

    AmrFunc func = ParseBlock(block_name);
    switch (func) {
      case AmrFunc::RedistributeAndRefine:
        MarkRedistributeEnd();
        break;
      case AmrFunc::SendBoundBuf:
        MarkSendBoundBufEnd();
        break;
      case AmrFunc::RecvBoundBuf:
        MarkRecvBoundBufEnd();
        break;
      case AmrFunc::SendFluxCor:
        MarkSendFluxCorEnd();
        break;
      case AmrFunc::RecvFluxCor:
        MarkRecvFluxCorEnd();
        break;
      case AmrFunc::MakeOutputs:
        MarkMakeOutputsEnd();
        break;
      default:
        break;
    }
  }

  void RegisterSend(uint64_t msg_tag, uint64_t dest, uint64_t msg_sz,
                    uint64_t timestamp) {
    if (rank_ == 0) {
      logf(LOG_DBUG, "SendMsg, Src: %" PRIu64 ", Dest: %" PRIu64, rank_, dest);
    }

    msglog_->LogMsg(dest, timestep_, PhaseToStr(), msg_tag, 0, msg_sz,
                    timestamp);
  }

  void RegisterRecv(uint64_t msg_tag, uint64_t src, uint64_t msg_sz,
                    uint64_t timestamp) {
    if (rank_ == 0) {
      logf(LOG_DBUG, "RecvMsg, Src: %" PRIu64 ", Dest: %" PRIu64, src, rank_);
    }

    msglog_->LogMsg(src, timestep_, PhaseToStr(), msg_tag, 1, msg_sz,
                    timestamp);
  }

  void ProcessTriggerMsg(void* data);

  void PrintStats() {
    if (rank_ == 0) {
      logf(LOG_INFO, "Num TimeSteps:\t %d", timestep_);
      logf(LOG_INFO, "Num Redistributions:\t %d", num_redistrib_);
    }
  }

 private:
  const char* PhaseToStr() const {
    switch (phase_) {
      case AMRPhase::FluxExchange:
        return "FluxExchange";
      case AMRPhase::LoadBalancing:
        return "LoadBalancing";
      case AMRPhase::BoundaryComm:
        return "BoundaryComm";
    }

    return "Unknown";
  }

  void MarkRedistributeBegin() {
    if (paranoid_) mutex_.lock();

    redistribute_ongoing_ = true;
    phase_ = AMRPhase::LoadBalancing;

    if (paranoid_) mutex_.unlock();
  }

  void MarkRedistributeEnd() {
    if (paranoid_) mutex_.lock();

    redistribute_ongoing_ = false;
    num_redistrib_++;

    if (paranoid_) mutex_.unlock();
  }

  void MarkSendBoundBufBegin() {
    if (paranoid_) mutex_.lock();

    if (!redistribute_ongoing_) {
      phase_ = AMRPhase::BoundaryComm;
    }

    if (paranoid_) mutex_.unlock();
  }

  void MarkSendBoundBufEnd() { /* noop */
  }

  void MarkRecvBoundBufBegin() { /* noop */
  }

  void MarkRecvBoundBufEnd() { /* noop */
  }

  void MarkSendFluxCorBegin() {
    if (paranoid_) mutex_.lock();

    if (!redistribute_ongoing_) {
      phase_ = AMRPhase::FluxExchange;
    }

    if (paranoid_) mutex_.unlock();
  }

  void MarkSendFluxCorEnd() { /* noop */
  }

  void MarkRecvFluxCorBegin() { /* noop */
  }

  void MarkRecvFluxCorEnd() { /* noop */
  }

  void MarkMakeOutputsBegin() { /* noop */
  }

  void MarkMakeOutputsEnd() { timestep_++; }

  void ProcessTriggerMsgBlockAssignment(void *data);

  void ProcessTriggerMsgTargetCost(void *data);

  void ProcessTriggerMsgBlockEvent(void *data);

  int rank_;
  int size_;

  int timestep_;
  int ts_sim_;
  int num_redistrib_;
  AMRPhase phase_;

  std::unique_ptr<MsgLog> msglog_;
  std::unique_ptr<FuncLog> funclog_;
  std::unique_ptr<StateLog> statelog_;
  std::unique_ptr<ProfLog> proflog_;

  bool redistribute_ongoing_;

  std::mutex mutex_;
  static const bool paranoid_ = false;
};

}
