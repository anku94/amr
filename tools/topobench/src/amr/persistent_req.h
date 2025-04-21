#pragma once

#include <mpi.h>

#include "amr_types.h"
#include "logging.h"

namespace topo {
class PersistentReq {
 public:
  // Non-copyable, but moveable
  PersistentReq(const PersistentReq&) = delete;
  PersistentReq& operator=(const PersistentReq&) = delete;
  PersistentReq(PersistentReq&& other) noexcept : req_(std::move(other.req_)) {}
  PersistentReq& operator=(PersistentReq&& other) noexcept {
    if (this != &other) {
      if (req_ != nullptr) {
        MPI_Request_free(req_.get());
        req_ = nullptr;
      }

      req_ = std::move(other.req_);
    }
    return *this;
  }

  void InitSend(void* buf, int count, MPI_Datatype datatype,
                                int dest, int tag, MPI_Comm comm) {
    int rv = MPI_Send_init(buf, count, datatype, dest, tag, comm, req_.get());
    ABORTIF(rv != MPI_SUCCESS, "MPI_Send_init failed");
  }

  void InitRecv(void* buf, int count, MPI_Datatype datatype,
                                int source, int tag, MPI_Comm comm) {
    int rv = MPI_Recv_init(buf, count, datatype, source, tag, comm, req_.get());
    ABORTIF(rv != MPI_SUCCESS, "MPI_Recv_init failed");
  }

  void Start() {
    ABORTIF(req_ == nullptr, "PersistentReq is not initialized");
    MPI_Start(req_.get());
  }

  void Wait() {
    ABORTIF(req_ == nullptr, "PersistentReq is not initialized");
    MPI_Wait(req_.get(), MPI_STATUS_IGNORE);
  }

  bool Test() {
    ABORTIF(req_ == nullptr, "PersistentReq is not initialized");
    int flag;
    MPI_Test(req_.get(), &flag, MPI_STATUS_IGNORE);
    return flag;
  }

  MpiReqUref Release() { return std::move(req_); }

  void Reset() {
    req_ = MpiReqUref(new MPI_Request(MPI_REQUEST_NULL), FreeMpiReq);
  }

  PersistentReq() : req_(MpiReqUref(new MPI_Request(MPI_REQUEST_NULL), FreeMpiReq)) {}

 private:

  static void FreeMpiReq(MPI_Request* req) {
    if (req != nullptr and *req != MPI_REQUEST_NULL) {
      MPI_Request_free(req);
      delete req;
    }
  }

  MpiReqUref req_;
};

}  // namespace topo