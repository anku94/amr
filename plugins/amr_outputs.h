#pragma once

class MsgLog {
 public:
  MsgLog(const char* dir, int rank) : rank_(rank), file_(nullptr) {
    const char* fname = "msgs";
    char fpath[4096];
    snprintf(fpath, 4096, "%s/%s.%d.csv", dir, fname, rank);
    file_ = fopen(fpath, "w+");
    if (file_ == nullptr) {
      ABORT("Failed to open CSV");
    }
    WriteHeader();
  }

  void LogMsg(int peer, int timestep, const char* phase, uint64_t msg_id,
              int send_or_recv, uint64_t msg_sz, uint64_t timestamp) {
    if (paranoid_) mutex_.lock();

    const char* fmt = "%d,%d,%d,%s,%" PRIu64 ",%d,%" PRIu64 ",%" PRIu64 "\n";
    fprintf(file_, fmt, rank_, peer, timestep, phase, msg_id, send_or_recv,
            msg_sz, timestamp);

    if (paranoid_) mutex_.unlock();
  }

 private:
  void WriteHeader() {
    const char* const header =
        "rank,peer,timestep,phase,msg_id,send_or_recv,msg_sz,timestamp\n";
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
    const char* fname = "funcs";
    char fpath[4096];
    snprintf(fpath, 4096, "%s/%s.%d.csv", dir, fname, rank);
    file_ = fopen(fpath, "w+");
    if (file_ == nullptr) {
      ABORT("Failed to open CSV");
    }
    WriteHeader();
  }

#define DONOTLOG(s) if (strncmp(func_name, s, strlen(s)) == 0) return;

  void LogFunc(const char* func_name, uint64_t timestamp, bool enter_or_exit) {
    // DONOTLOG("Task_ReceiveBoundaryBuffers_MeshBlockData");
    // DONOTLOG("Task_ReceiveBoundaryBuffers_MeshData");
    DONOTLOG("Task_ReceiveBoundaryBuffers_Mesh");

    const char* fmt = "%d,%" PRIu64 ",%s,%c\n";
    fprintf(file_, fmt, rank_, timestamp, func_name, enter_or_exit ? '0' : '1');
  }

 private:
  void WriteHeader() {
    const char* const header = "rank,timestep,func,enter_or_exit\n";
    fprintf(file_, header);
  }

  int rank_;
  FILE* file_;

  std::mutex mutex_;
  static const bool paranoid_ = true;
};
