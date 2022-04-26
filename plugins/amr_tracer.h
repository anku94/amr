#include "../tools/common.cc"
#include "../tools/common.h"
#include "amr_util.h"

#include <inttypes.h>
#include <mpi/mpi.h>
#include <mutex>

enum class AMRPhase { FluxExchange, LoadBalancing, BoundaryComm };

class AMRTracer {
 public:
  AMRTracer()
      : rank_(-1),
        size_(0),
        timestep_(0),
        num_redistrib_(0),
        phase_(AMRPhase::LoadBalancing),
        csv_out_(nullptr),
        redistribute_ongoing_(false) {
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    PMPI_Comm_size(MPI_COMM_WORLD, &size_);

    const char* dir = "/mnt/lt20ad2/parthenon-topo/profile";
    char fpath[1024];
    snprintf(fpath, 1024, "%s/log.%d.csv", dir, rank_);

    csv_out_ = fopen(fpath, "w+");

    if (csv_out_ == nullptr) {
      ABORT("Failed to open CSV");
    }

    LogCSVHeader();
  }

  int MyRank() const { return rank_; }

#define DEFINE_BLOCK(s)                              \
  void Mark##s(const char* block_name) {             \
    amr::AmrFunc func = amr::ParseBlock(block_name); \
    switch (func) {                                  \
      case amr::AmrFunc::RedistributeAndRefine:      \
        MarkRedistribute##s();                       \
        break;                                       \
      case amr::AmrFunc::SendBoundBuf:               \
        MarkSendBoundBuf##s();                       \
        break;                                       \
      case amr::AmrFunc::RecvBoundBuf:               \
        MarkRecvBoundBuf##s();                       \
        break;                                       \
      case amr::AmrFunc::SendFluxCor:                \
        MarkSendFluxCor##s();                        \
        break;                                       \
      case amr::AmrFunc::RecvFluxCor:                \
        MarkRecvFluxCor##s();                        \
        break;                                       \
      case amr::AmrFunc::MakeOutputs:                \
        MarkMakeOutputs##s();                        \
        break;                                       \
      default:                                       \
        break;                                       \
    }                                                \
  }

  DEFINE_BLOCK(Begin)
  DEFINE_BLOCK(End)

  // void MarkBegin(const char* block_name) {
  // amr::AmrFunc func = amr::ParseBlock(block_name);
  // switch(func) {
  // case amr::AmrFunc::RedistributeAndRefine:
  // MarkRedistributeBegin();
  // break;
  // case amr::AmrFunc::MakeOutputs:
  // MarkMakeOutputsBegin();
  // break;
  // default:
  // break;
  // }
  // }

  // void MarkEnd(const char* block_name) {
  // amr::AmrFunc func = amr::ParseBlock(block_name);
  // switch(func) {
  // case amr::AmrFunc::RedistributeAndRefine:
  // MarkRedistributeEnd();
  // break;
  // case amr::AmrFunc::MakeOutputs:
  // MarkMakeOutputsBegin();
  // break;
  // default:
  // break;
  // }
  // }

  void RegisterSend(uint64_t msg_tag, uint64_t dest, uint64_t msg_sz,
                    uint64_t timestamp) {
    LogCSV(dest, msg_tag, 0, msg_sz, timestamp);
  }

  void RegisterRecv(uint64_t msg_tag, uint64_t src, uint64_t msg_sz,
                    uint64_t timestamp) {
    LogCSV(src, msg_tag, 1, msg_sz, timestamp);
  }

  ~AMRTracer() {
    if (csv_out_ != nullptr) {
      fclose(csv_out_);
      csv_out_ = nullptr;
    }
  }

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

  void LogCSVHeader() {
    const char* const header =
        "rank,peer,timestep,phase,msg_id,send_or_recv,msg_sz,timestamp\n";
    fprintf(csv_out_, header);
  }

  void LogCSV(uint64_t peer, uint64_t msg_id, int send_or_recv, uint64_t msg_sz,
              uint64_t timestamp) {
    if (paranoid_) mutex_.lock();

    const char* fmt = "%d,%d,%d,%s,%" PRIu64 ",%d,%" PRIu64 ",%" PRIu64 "\n";
    fprintf(csv_out_, fmt, rank_, peer, timestep_, /* phase */ PhaseToStr(),
            msg_id, send_or_recv, msg_sz, timestamp);

    if (paranoid_) mutex_.unlock();
  }

  int rank_;
  int size_;

  int timestep_;
  int num_redistrib_;
  AMRPhase phase_;

  FILE* csv_out_;

  bool redistribute_ongoing_;

  std::mutex mutex_;
  static const bool paranoid_ = false;
};
