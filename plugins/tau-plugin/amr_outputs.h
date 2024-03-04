#pragma once

#include "amr_util.h"

#include <inttypes.h>
#include <mutex>

namespace tau {
class MsgLog {
 public:
  MsgLog(const char* dir, int rank) : rank_(rank), file_(nullptr) {
    EnsureFileOrDie(&file_, dir, "msgs", "bin", rank);
  }

  ~MsgLog() {
    if (file_) {
      fclose(file_);
      file_ = nullptr;
    }
  }

  void LogChannel(void* ptr, int block_id, int rank, int nbr_id, int nbr_rank,
                  int tag, char is_flux) {
    static constexpr size_t recsz = 8 + 5 * sizeof(int) + sizeof(char);
    channels_.resize(channels_.size() + recsz);

    char* bufptr = channels_.data() + channels_.size() - recsz;

    memcpy(bufptr, ptr, 8);
    bufptr += 8;

    memcpy(bufptr, (void*)&block_id, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&rank, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&nbr_id, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&nbr_rank, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&tag, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&is_flux, sizeof(char));
    bufptr += sizeof(char);
  }

  void LogSend(void* ptr, int buf_sz, int recv_rank, int tag,
               uint64_t timestamp) {
    // uint64_t time_us = GetTimestampUs();
    static constexpr size_t recsz = 16 + 3 * sizeof(int);
    sends_.resize(sends_.size() + recsz);

    char* bufptr = sends_.data() + sends_.size() - recsz;

    memcpy(bufptr, ptr, 8);
    bufptr += 8;

    memcpy(bufptr, (void*)&buf_sz, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&recv_rank, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&tag, sizeof(int));
    bufptr += sizeof(int);

    memcpy(bufptr, (void*)&timestamp, 8);
    bufptr += 8;
  }

  void Flush(int ts) {
    fwrite(&ts, sizeof(int), 1, file_);

    int channelsz = channels_.size();
    fwrite(&channelsz, sizeof(int), 1, file_);
    if (channelsz > 0) {
      fwrite(channels_.data(), channels_.size(), 1, file_);
    }
    channels_.clear();

    int sendsz = sends_.size();
    fwrite(&sendsz, sizeof(int), 1, file_);
    if (sendsz > 0) {
      fwrite(sends_.data(), sends_.size(), 1, file_);
    }
    sends_.clear();
  }

 private:
  uint64_t GetTimestampUs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    uint64_t time_us = (ts.tv_sec * 1e6) + (ts.tv_nsec * 1e-3);
    return time_us;
  }

  int rank_;
  FILE* file_;
  std::vector<char> channels_;
  std::vector<char> sends_;
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

  void LogEvent(int ts, int block_id, int opcode, int data) {
    fwrite(&ts, sizeof(int), 1, file_);
    fwrite(&block_id, sizeof(int), 1, file_);
    fwrite(&opcode, sizeof(int), 1, file_);
    fwrite(&data, sizeof(int), 1, file_);
  }

  ~ProfLog() { fclose(file_); }

 private:
  int rank_;
  FILE* file_;
};
}  // namespace tau
