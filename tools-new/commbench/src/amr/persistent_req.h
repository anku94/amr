#pragma once

#include <mpi.h>

#include "amr_types.h"
#include "tools-common/logging.h"

namespace topo {
class PersistentReq {
 public:
  // Non-copyable, but moveable
  PersistentReq(const PersistentReq&) = delete;
  PersistentReq& operator=(const PersistentReq&) = delete;
  PersistentReq(PersistentReq&& other) noexcept
      : req_(std::move(other.req_)), completed_(other.completed_) {
    other.completed_ = true;
  }
  PersistentReq& operator=(PersistentReq&& other) noexcept {
    if (this != &other) {
      req_ = std::move(other.req_);
      completed_ = other.completed_;
      other.completed_ = true;
    }
    return *this;
  }

  void InitSend(void* buf, int count, MPI_Datatype datatype, int dest, int tag,
                MPI_Comm comm) {
    Reset();
    int rv = MPI_Send_init(buf, count, datatype, dest, tag, comm, req_.get());
    ABORTIF(rv != MPI_SUCCESS, "MPI_Send_init failed");
  }

  void InitRecv(void* buf, int count, MPI_Datatype datatype, int source,
                int tag, MPI_Comm comm) {
    Reset();
    int rv = MPI_Recv_init(buf, count, datatype, source, tag, comm, req_.get());
    ABORTIF(rv != MPI_SUCCESS, "MPI_Recv_init failed");
  }

  // Start: start the request
  void Start() {
    ABORTIF(req_ == nullptr, "PersistentReq is not initialized");
    MPI_Start(req_.get());
    completed_ = false;
  }

  // Wait: wait for request to complete
  void Wait() {
    ABORTIF(req_ == nullptr, "PersistentReq is not initialized");
    MPI_Status status;
    int rv = MPI_Wait(req_.get(), &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI_Wait failed");
    completed_ = true;
  }

  // Test: check if request is done
  bool Test() {
    if (completed_) {
      return true;
    }

    ABORTIF(req_ == nullptr, "PersistentReq is not initialized");
    int flag;
    MPI_Status status;
    int rv = MPI_Test(req_.get(), &flag, &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI_Test failed");

    if (flag) {
      completed_ = true;
    }

    return flag;
  }

  // Release: release ownership of mpi_request
  MpiReqUref Release() {
    completed_ = true;
    return std::move(req_);
  }

  PersistentReq()
      : req_(MpiReqUref(new MPI_Request(MPI_REQUEST_NULL), FreeMpiReq)),
        completed_{true} {}

  ~PersistentReq() {
    if (req_ != nullptr && *req_ != MPI_REQUEST_NULL && !completed_) {
      MLOGIFR0(MLOG_WARN, "Destroying incomplete request");
      Wait();
    }
  }

 private:
  // Reset: free old req, allocate new one
  void Reset() {
    if (req_ != nullptr && *req_ != MPI_REQUEST_NULL && !completed_) {
      MLOGIFR0(MLOG_WARN, "Resetting incomplete request");
      Wait();
    }
    req_ = MpiReqUref(new MPI_Request(MPI_REQUEST_NULL), FreeMpiReq);
  }

  // FreeMpiReq: util fn for MpiReqUref deleter
  static void FreeMpiReq(MPI_Request* req) {
    if (req != nullptr) {
      if (*req != MPI_REQUEST_NULL) {
        MPI_Request_free(req);
      }
      delete req;
    }
  }

  MpiReqUref req_;
  bool completed_;
};

}  // namespace topo