#include "logging.h"

#include <ctime>
#include <mpi.h>
#include <unordered_map>

namespace amr {
class Tracer {
 public:
  Tracer(pdlfs::WritableFile* fout, int rank) : fout_(fout), rank_(rank) {
    logvat0(__LOG_ARGS__, LOG_INFO, "Tracer initializing on rank %d:%p", rank_,
            fout_)
  }

  static double GetNowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
    return now;
  }

  void LogFuncBegin(const char* func_name) {
    if (!fout_) {
      return;
    }

    char buf[1024];
    int bufwr =
        snprintf(buf, sizeof(buf), "\"%s\",0,%lf\n", func_name, GetNowMs());

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogFuncEnd(const char* func_name) {
    if (!fout_) {
      return;
    }

    char buf[1024];
    int bufwr =
        snprintf(buf, sizeof(buf), "\"%s\",1,%lf\n", func_name, GetNowMs());

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogInner(const char* func_name, int is_closed, double ts) {
    if (!fout_) {
      return;
    }

    char buf[1024];
    int bufwr =
      snprintf(buf, sizeof(buf), "\"%s\",%d,%lf\n", func_name, is_closed, ts);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }


  //
  // Used for coalescing, writes prev func from maps and clears it
  //
  void FlushPrev() {
    if (!fout_) {
      return;
    }

    if (prev_ == "") {
      return;
    }

    // for nested calls, may find both start and end of prev
    // or only start or only end
    if (func_start_.find(prev_) != func_start_.end()) {
      LogInner(prev_.c_str(), 0, func_start_[prev_]);
      func_start_.erase(prev_);
    }

    if (func_end_.find(prev_) != func_end_.end()) {
      LogInner(prev_.c_str(), 1, func_end_[prev_]);
      func_end_.erase(prev_);
    }

    prev_ = "";
  }

  void LogFuncBeginCoalesced(const char* func_name) {
    if (!fout_) return;

    if (prev_ != "" and prev_ != func_name) {
      FlushPrev();
    }

    prev_ = func_name;

    // if we're coalescing, first call should be logged
    if (func_start_.find(func_name) == func_start_.end()) {
      func_start_[func_name] = GetNowMs();
    }
  }

  void LogFuncEndCoalesced(const char* func_name) {
    if (!fout_) return;

    if (prev_ != "" and prev_ != func_name) {
      FlushPrev();
    }

    prev_ = func_name;

    // if we're coalescing, last end should be logged
    func_end_[func_name] = GetNowMs();
  }

  void LogMPIIsend(void* reqptr, int count, int dest, int tag) {
    if (!fout_) {
      return;
    }

    return;

    char buf[1024];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Isend,%lf,%p,%d,%d,%d\n",
                         GetNowMs(), reqptr, count, dest, tag);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void MPIIrecv(void* reqptr, int count, int source, int tag) {
    if (!fout_) {
      return;
    }

    return;

    char buf[1024];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Irecv,%lf,%p,%d,%d,%d\n",
                         GetNowMs(), reqptr, count, source, tag);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogMPITestEnd(void* reqptr, int flag) {
    if (!fout_) {
      return;
    }

    return;

    char buf[1024];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Test,%lf,%p,%d\n", GetNowMs(),
                         reqptr, flag);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogMPIWait(void* reqptr) {
    if (!fout_) {
      return;
    }

    return;

    char buf[1024];
    int bufwr =
        snprintf(buf, sizeof(buf), "MPI_Wait,%lf,%p\n", GetNowMs(), reqptr);
fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogMPIWaitall(MPI_Request* reqptr, int count) {
    if (!fout_) {
      return;
    }

    int bufsz = 1024 + 16 * count;
    char buf[bufsz];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Waitall,%lf,%p,%d\n",
                         GetNowMs(), reqptr, count);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  ~Tracer() {
    logvat0(__LOG_ARGS__, LOG_DBUG, "Tracer destroyed on rank %d", rank_);

    if (fout_ != nullptr) {
      FlushPrev();
      fout_->Flush();
      fout_->Close();
      fout_ = nullptr;
    }
  }

 private:
  // for coalescing consecutive calls
  std::unordered_map<std::string, double> func_start_;
  std::unordered_map<std::string, double> func_end_;
  std::string prev_;

  pdlfs::WritableFile* fout_;
  int rank_;
};
}  // namespace amr
