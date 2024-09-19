#include "logging.h"

#include <ctime>
#include <mpi.h>

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
    int bufwr = snprintf(buf, sizeof(buf), "%s,%lf\n", func_name, GetNowMs());

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogMPIIsend(void* reqptr, int count, int dest, int tag) {
    if (!fout_) {
      return;
    }

    char buf[1024];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Isend,%lf,%p,%d,%d,%d\n",
                         GetNowMs(), reqptr, count, dest, tag);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void MPIIrecv(void* reqptr, int count, int source, int tag) {
    if (!fout_) {
      return;
    }

    char buf[1024];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Irecv,%lf,%p,%d,%d,%d\n",
                         GetNowMs(), reqptr, count, source, tag);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogMPITestEnd(void* reqptr, int flag) {
    if (!fout_) {
      return;
    }

    char buf[1024];
    int bufwr = snprintf(buf, sizeof(buf), "MPI_Test,%lf,%p,%d\n", GetNowMs(),
                         reqptr, flag);

    fout_->Append(pdlfs::Slice(buf, bufwr));
  }

  void LogMPIWait(void* reqptr) {
    if (!fout_) {
      return;
    }

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
      fout_->Flush();
      fout_->Close();
      fout_ = nullptr;
    }
  }

 private:
  pdlfs::WritableFile* fout_;
  int rank_;
};
}  // namespace amr
