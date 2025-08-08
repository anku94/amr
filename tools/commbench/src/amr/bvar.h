//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include <mpi.h>

#include <memory>

#include "amr_types.h"
#include "drain_queue.h"
#include "persistent_req.h"
#include "tools-common/globals.h"
#include "tools-common/logging.h"

#define NMAX_NEIGHBORS 512
#define MAX_MSGSZ 32768

namespace topo {
class MeshBlock;

//
// BoundaryStatus:
// - waiting: initial status
// - arrived: after MPI_Wait
// - completed: notused?
//
enum class BoundaryStatus { waiting, arrived, completed };

/* kMaxNeighbor should be set to 2X the actual possible value,
 * as we don't share BoundaryData indexes for sends and receives
 * to the same neighbor
 */

//
// BoundaryData: preallocated buffers for MPI comm
// - Held by BoundaryVariable (which is held by MeshBlock)
// - currently one bvar holds one instance for flxcorr and one for BC
// - NeighborBlock buf_id curresponds to a single entry
// - Also read the comment above, don't remember why we need 2x
//
template <int n = NMAX_NEIGHBORS>
struct BoundaryData {
  static constexpr int kMaxNeighbor = n;
  BoundaryStatus flag[kMaxNeighbor], sflag[kMaxNeighbor];
  char sendbuf[kMaxNeighbor][MAX_MSGSZ];  // used by MPI_Send
  char recvbuf[kMaxNeighbor][MAX_MSGSZ];
  int sendbufsz[kMaxNeighbor];
  int recvbufsz[kMaxNeighbor];
  PersistentReq req_send[kMaxNeighbor], req_recv[kMaxNeighbor];
};

//
// BoundaryVariable: orchestrates MPI comm using boundarydata
// on behalf of a single MeshBlock. Mirrors Parthenon interface.
//
class BoundaryVariable {
 public:
  BoundaryVariable(std::weak_ptr<MeshBlock> wpmb)
      : wpmb_(wpmb), bytes_sent_(0), bytes_rcvd_(0) {
    InitBoundaryData(bd_var_);
    InitBoundaryData(bd_var_flcor_);
  }

  // InitBoundaryData: initialize bd struct (status: waiting, req: null)
  // Does not do any allocation
  void InitBoundaryData(BoundaryData<>& bd);

  // SetupPersistentMPI: allocate MPI_Request arrays
  void SetupPersistentMPI();

  // StartReceiving: post non-blocking MPI_Irecv
  void StartReceiving();

  // ClearBoundary: reaps sends next time around // should change?
  void ClearBoundary();

  // SendBoundaryBuffers: trigger MPI_Start for sends
  void SendBoundaryBuffers();

  // ReceiveBoundaryBuffers: call iprobe and mpi_test (non-blocking)
  // needs to be called in a while loop until it returns true to progress
  bool ReceiveBoundaryBuffers();

  // ReceiveBoundaryBuffersWithWait: blocking variant, calls MPI_Wait
  // in order of buf_id
  void ReceiveBoundaryBuffersWithWait();

  // DestroyBoundaryData: destroy MPI requests in bd and set to null
  // reverse of SetupPersistentMPI
  void DestroyBoundaryData(BoundaryData<>& bd);

  ~BoundaryVariable() {
    DestroyBoundaryData(bd_var_);
    DestroyBoundaryData(bd_var_flcor_);
  }

 private:
  friend class MeshBlock;
  SendDrainQueue dq_;  // Drain queue for incomplete sends

  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (wpmb_.expired()) {
      ABORT("Invalid pointer to MeshBlock!");
    }

    return wpmb_.lock();
  }

  std::weak_ptr<MeshBlock> wpmb_;
  BoundaryData<> bd_var_, bd_var_flcor_;
  uint64_t bytes_sent_, bytes_rcvd_;  // msg sizes
  uint64_t sendcnt_{0}, recvcnt_{0};  // msg count

  static constexpr int kMaxNeighbor = NMAX_NEIGHBORS;
  static constexpr int kMaxMsgSz = MAX_MSGSZ;
};
}  // namespace topo
