//
// Created by Ankush J on 4/11/22.
//

#include "block.h"
#include "drain_queue.h"
#include "tools-common/globals.h"

namespace topo {
void BoundaryVariable::InitBoundaryData(BoundaryData<>& bd) {
  for (int n = 0; n < bd.kMaxNeighbor; n++) {
    bd.flag[n] = BoundaryStatus::waiting;
    bd.sflag[n] = BoundaryStatus::waiting;
  }
}

void BoundaryVariable::SetupPersistentMPI() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    ABORTIF(nb.buf_id >= kMaxNeighbor, "buf_id out of bounds");
    ABORTIF(nb.msg_sz >= kMaxMsgSz, "msg_sz out of bounds");

    MLOGIFR0(MLOG_DBG1, "[SEND] Us: %d, Neighbor: %d (bufid: %d)",
             Globals::my_rank, nb.peer_rank, nb.buf_id);

    auto& preq = bd_var_.req_send[nb.buf_id];
    // buffer, msgsize, datatype, dest, tag, comm, req
    preq.InitSend(bd_var_.sendbuf[nb.buf_id], nb.msg_sz, MPI_CHAR, nb.peer_rank,
                  nb.tag, MPI_COMM_WORLD);
    bd_var_.sendbufsz[nb.buf_id] = nb.msg_sz;
  }

  for (auto nb : pmb->nbrvec_rcv_) {
    MLOGIFR0(MLOG_DBG1, "[RECV] Us: %d, Neighbor: %d (bufid: %d)",
             Globals::my_rank, nb.peer_rank, nb.buf_id);

    auto& preq = bd_var_.req_recv[nb.buf_id];
    preq.InitRecv(bd_var_.recvbuf[nb.buf_id], nb.msg_sz, MPI_CHAR, nb.peer_rank,
                  nb.tag, MPI_COMM_WORLD);
    bd_var_.recvbufsz[nb.buf_id] = nb.msg_sz;
  }
}

void BoundaryVariable::StartReceiving() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_rcv_) {
    bd_var_.req_recv[nb.buf_id].Start();

    MLOGIFR0(MLOG_DBG2, "Rank %d - Receive POSTED %d", Globals::my_rank,
             nb.buf_id);
  }
}

void BoundaryVariable::ClearBoundary() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    bd_var_.flag[nb.buf_id] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.buf_id] = BoundaryStatus::waiting;
    // bd_var_.req_send[nb.buf_id].Wait();
    bytes_sent_ += bd_var_.sendbufsz[nb.buf_id];
  }

  for (auto nb : pmb->nbrvec_rcv_) {
    bd_var_.flag[nb.buf_id] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.buf_id] = BoundaryStatus::waiting;
  }
}

void BoundaryVariable::SendBoundaryBuffers() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    auto& preq = bd_var_.req_send[nb.buf_id];
    // Move incomplete requests to drain queue
    if (!preq.Test()) {
      // Still incomplete
      dq_.AddSendRequest(preq.Release());
      preq.InitSend(bd_var_.sendbuf[nb.buf_id], bd_var_.sendbufsz[nb.buf_id],
                    MPI_CHAR, nb.peer_rank, nb.tag, MPI_COMM_WORLD);
    }

    MLOGIFR0(MLOG_DBG1, "R%d - Send START %d", Globals::my_rank, nb.buf_id);
    bd_var_.req_send[nb.buf_id].Start();
    MLOGIFR0(MLOG_DBG1, "R%d - Send POSTED %d", Globals::my_rank, nb.buf_id);
    sendcnt_++;
  }
}

bool BoundaryVariable::ReceiveBoundaryBuffers() {
  bool bflag = true;
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MPI_Status status;

  for (auto nb : pmb->nbrvec_rcv_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    int test;
    int rv =
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI Iprobe Failed");

    bool done = bd_var_.req_recv[nb.buf_id].Test();
    if (!done) {
      bflag = false;
      continue;
    }

    bd_var_.flag[nb.buf_id] = BoundaryStatus::arrived;
    bytes_rcvd_ += bd_var_.recvbufsz[nb.buf_id];
    recvcnt_++;
  }

  return bflag;
}

void BoundaryVariable::ReceiveBoundaryBuffersWithWait() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MPI_Status status;

  for (auto nb : pmb->nbrvec_rcv_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    bd_var_.req_recv[nb.buf_id].Wait();

    // redundant; guaranteed with MPI_Wait
    bd_var_.flag[nb.buf_id] = BoundaryStatus::arrived;
    bytes_rcvd_ += bd_var_.recvbufsz[nb.buf_id];
    recvcnt_++;
  }

  dq_.TryDraining();
}

void BoundaryVariable::DestroyBoundaryData(BoundaryData<>& bd) {
  MLOGIFR0(MLOG_DBG2, "Destroying boundary data");
  // We do not explicitly destroy now as PersistentReq is RAII-safe
}
}  // namespace topo
