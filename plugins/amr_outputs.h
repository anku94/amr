#pragma once

#include <inttypes.h>
#include <mutex>

#include "amr_util.h"

namespace tau {

class MsgLog {
 public:
  MsgLog(const char* dir, int rank) : rank_(rank), file_(nullptr) {
    EnsureFileOrDie(&file_, dir, "msgs", "csv", rank_);
    WriteHeader();
  }

  void LogMsg(int peer, int timestep, const char* phase, uint64_t msg_id,
              int send_or_recv, uint64_t msg_sz, uint64_t timestamp) {
    if (paranoid_) mutex_.lock();

    const char* phase_mapped = MapPhaseName(phase);

    const char* fmt = "%d|%d|%d|%s|%" PRIu64 "|%d|%" PRIu64 "|%" PRIu64 "\n";
    fprintf(file_, fmt, rank_, peer, timestep, phase_mapped, msg_id,
            send_or_recv, msg_sz, timestamp);

    if (paranoid_) mutex_.unlock();
  }

 private:
  const char* MapPhaseName(const char* phase_name) {
#define PHASE_IS(x) (strncmp(phase_name, x, strlen(x)) == 0)
    if (PHASE_IS("FluxExchange")) {
      return "FE";
    } else if (PHASE_IS("BoundaryComm")) {
      return "BC";
    } else if (PHASE_IS("LoadBalancing")) {
      return "LB";
    } else {
      return phase_name;
    }
  }

  void WriteHeader() {
    const char* const header =
        "rank|peer|timestep|phase|msg_id|send_or_recv|msg_sz|timestamp\n";
    fprintf(file_, header);
  }

  int rank_;
  FILE* file_;

  std::mutex mutex_;
  static const bool paranoid_ = false;
};

class FuncLog {
 public:
  FuncLog(const char* dir, int rank) : rank_(rank), file_(nullptr) {
    EnsureFileOrDie(&file_, dir, "funcs", "csv", rank);
    WriteHeader();
  }

#define DONOTLOG(s) \
  if (strncmp(func_name, s, strlen(s)) == 0) return;

  void LogFunc(const char* func_name, int timestep, uint64_t timestamp,
               bool enter_or_exit) {
    // DONOTLOG("Task_ReceiveBoundaryBuffers_MeshBlockData");
    // DONOTLOG("Task_ReceiveBoundaryBuffers_MeshData");
    DONOTLOG("Task_ReceiveBoundaryBuffers_Mesh");

    const char* fmt = "%d|%d|%" PRIu64 "|%s|%c\n";
    fprintf(file_, fmt, rank_, timestep, timestamp, func_name,
            enter_or_exit ? '0' : '1');
  }

 private:
  void WriteHeader() {
    const char* const header = "rank|timestep|timestamp|func|enter_or_exit\n";
    fprintf(file_, header);
  }

  int rank_;
  FILE* file_;

  std::mutex mutex_;
  static const bool paranoid_ = true;
};

class StateLog {
 public:
  StateLog(const char* dir, int rank) : rank_(rank), file_(nullptr) {
    EnsureFileOrDie(&file_, dir, "state", "csv", rank);
    WriteHeader();
  }

  void LogKV(int timestep, const char* key, const char* val) {
    const char* fmt = "%d|%d|%s|%s\n";
    fprintf(file_, fmt, rank_, timestep, key, val);
  }

 private:
  void WriteHeader() {
    const char* const header = "rank|timestep|key|val\n";
    fprintf(file_, header);
  }

  int rank_;
  FILE* file_;

  std::mutex mutex_;
  static const bool paranoid_ = true;
};

class ProfLog {
  public:
    ProfLog(const char* dir, int rank) : rank_(rank), file_(nullptr) {
      EnsureFileOrDie(&file_, dir, "prof", "bin", rank);
    }

    void LogEvent(int ts, int block_id, int event_opcode, int event_us) {
      fwrite(&ts, sizeof(int), 1, file_);
      fwrite(&block_id, sizeof(int), 1, file_);
      fwrite(&event_opcode, sizeof(int), 1, file_);
      fwrite(&event_us, sizeof(int), 1, file_);
    }

    ~ProfLog() {
      fclose(file_);
    }

  private:
    int rank_;
    FILE* file_;
};
}  // namespace tau
