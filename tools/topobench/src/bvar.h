//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include "common.h"

#include <memory>
#include <mpi.h>

#define NMAX_NEIGHBORS 2048
#define MAX_MSGSZ 16384

class MeshBlock;

enum class BoundaryStatus { waiting, arrived, completed };

/* kMaxNeighbor should be set to 2X the actual possible value,
 * as we don't share BoundaryData indexes for sends and receives
 * to the same neighbor
 */
template <int n = NMAX_NEIGHBORS>
struct BoundaryData {
  static constexpr int kMaxNeighbor = n;
  BoundaryStatus flag[kMaxNeighbor], sflag[kMaxNeighbor];
  char sendbuf[kMaxNeighbor][MAX_MSGSZ];
  char recvbuf[kMaxNeighbor][MAX_MSGSZ];
  int sendbufsz[kMaxNeighbor];
  int recvbufsz[kMaxNeighbor];
  MPI_Request req_send[kMaxNeighbor], req_recv[kMaxNeighbor];
};

class BoundaryVariable {
 public:
  BoundaryVariable(std::weak_ptr<MeshBlock> wpmb)
      : wpmb_(wpmb), bytes_sent_(0), bytes_rcvd_(0) {
    InitBoundaryData(bd_var_);
    InitBoundaryData(bd_var_flcor_);
  }

  void InitBoundaryData(BoundaryData<>& bd);
  void SetupPersistentMPI();
  void StartReceiving();
  void ClearBoundary();
  void SendBoundaryBuffers();
  // needs to be called in a while loop until it returns true
  bool ReceiveBoundaryBuffers();
  void ReceiveBoundaryBuffersWithWait();
  void DestroyBoundaryData(BoundaryData<>& bd);

 private:
  friend class MeshBlock;
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (wpmb_.expired()) {
      ABORT("Invalid pointer to MeshBlock!");
    }

    return wpmb_.lock();
  }

  std::weak_ptr<MeshBlock> wpmb_;
  BoundaryData<> bd_var_, bd_var_flcor_;
  uint64_t bytes_sent_, bytes_rcvd_;
};
