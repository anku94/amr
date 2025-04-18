//
// Created by Ankush J on 4/11/22.
//

#include "block.h"
#include "globals.h"

#define MPI_CHECK_STATUS(rv, mpi_status, msg) \
  if (rv != MPI_SUCCESS) {                    \
    CheckMpiStatus(rv, mpi_status, msg);      \
  }

void CheckMpiStatus(int rv, int mpi_status, const char* msg) {
  if (rv == MPI_SUCCESS) return;

  char errbuf[MPI_MAX_ERROR_STRING];
  int errlen;
  MPI_Error_string(mpi_status, errbuf, &errlen);
  MLOG(MLOG_ERRO, "%s: %s", msg, errbuf);
  MPI_Abort(MPI_COMM_WORLD, mpi_status);
}

namespace topo {
void BoundaryVariable::InitBoundaryData(BoundaryData<>& bd) {
  for (int n = 0; n < bd.kMaxNeighbor; n++) {
    bd.flag[n] = BoundaryStatus::waiting;
    bd.sflag[n] = BoundaryStatus::waiting;
    bd.req_send[n] = MPI_REQUEST_NULL;
    bd.req_recv[n] = MPI_REQUEST_NULL;
  }
}

/* XXX: changing MPI_CHAR to MPI_BYTE because we were getting
 * ABORTs with MPI_CHAR, and quick-and-dirty search suggested this
 * to be appropriate.
 *
 * frame #2: 0x0000000100030a6c
 * meshdriver_main`topo::BoundaryVariable::ReceiveBoundaryBuffers(this=0x0000000148000000)
 * at bvar.cc:129:5 126      rv = MPI_Test(&(bd_var_.req_recv[nb.buf_id]),
 * &test, &status); 127 128      MPI_CHECK_STATUS(rv, "MPI Test Failed");
 * -> 129      MPI_CHECK_STATUS(status.MPI_ERROR, "MPI Test Failed");
 *  130
 *  131      if (!static_cast<bool>(test)) {
 *  132        bflag = false;
 * (lldb) p status
 * (MPI_Status) {
 *   count_lo = 20480
 *   count_hi_and_cancelled = 0
 *   MPI_SOURCE = 0
 *   MPI_TAG = 1
 *   MPI_ERROR = 5456768
 */

void BoundaryVariable::SetupPersistentMPI() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    ABORTIF(nb.buf_id >= kMaxNeighbor, "buf_id out of bounds");
    ABORTIF(nb.msg_sz >= kMaxMsgSz, "msg_sz out of bounds");

    MLOGIFR0(MLOG_DBG1, "[SEND] Us: %d, Neighbor: %d (bufid: %d)",
             Globals::my_rank, nb.peer_rank, nb.buf_id);

    if (bd_var_.req_send[nb.buf_id] != MPI_REQUEST_NULL)
      MPI_Request_free(&bd_var_.req_send[nb.buf_id]);

    // buffer, msgsize, datatype, dest, tag, comm, req
    MPI_Send_init(bd_var_.sendbuf[nb.buf_id], nb.msg_sz, MPI_CHAR, nb.peer_rank,
                  nb.tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.buf_id]));
    bd_var_.sendbufsz[nb.buf_id] = nb.msg_sz;
  }

  for (auto nb : pmb->nbrvec_rcv_) {
    MLOGIFR0(MLOG_DBG1, "[RECV] Us: %d, Neighbor: %d (bufid: %d)",
             Globals::my_rank, nb.peer_rank, nb.buf_id);

    if (bd_var_.req_recv[nb.buf_id] != MPI_REQUEST_NULL)
      MPI_Request_free(&bd_var_.req_recv[nb.buf_id]);

    MPI_Recv_init(bd_var_.recvbuf[nb.buf_id], nb.msg_sz, MPI_CHAR, nb.peer_rank,
                  nb.tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.buf_id]));
    bd_var_.recvbufsz[nb.buf_id] = nb.msg_sz;
  }
}

void BoundaryVariable::StartReceiving() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_rcv_) {
    int rv = MPI_Start(&(bd_var_.req_recv[nb.buf_id]));
    ABORTIF(rv != MPI_SUCCESS, "MPI Start Failed");

    MLOGIFR0(MLOG_DBG2, "Rank %d - Receive POSTED %d", Globals::my_rank,
             nb.buf_id);
  }
}

void BoundaryVariable::ClearBoundary() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    bd_var_.flag[nb.buf_id] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.buf_id] = BoundaryStatus::waiting;

    // XXX: some sort of fence
    MPI_Status status;
    int rv = MPI_Wait(&(bd_var_.req_send[nb.buf_id]), &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI Wait Failed");

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
    // some fence
    MLOGIFR0(MLOG_DBG1, "Rank %d - Send START %d", Globals::my_rank, nb.buf_id);

    int rv = MPI_Start(&(bd_var_.req_send[nb.buf_id]));
    ABORTIF(rv != MPI_SUCCESS, "MPI Start Failed");

    MLOGIFR0(MLOG_DBG1, "Rank %d - Send POSTED %d", Globals::my_rank,
             nb.buf_id);
  }
}

bool BoundaryVariable::ReceiveBoundaryBuffers() {
  bool bflag = true;
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MPI_Status status;

  for (auto nb : pmb->nbrvec_rcv_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    int test;
    int rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                        &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI Iprobe Failed");

    MPI_Status status;
    rv = MPI_Test(&(bd_var_.req_recv[nb.buf_id]), &test, &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI Test Failed");

    if (!static_cast<bool>(test)) {
      bflag = false;
      continue;
    }

    bd_var_.flag[nb.buf_id] = BoundaryStatus::arrived;
    bytes_rcvd_ += bd_var_.recvbufsz[nb.buf_id];
  }

  return bflag;
}

void BoundaryVariable::ReceiveBoundaryBuffersWithWait() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MPI_Status status;

  for (auto nb : pmb->nbrvec_rcv_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    int rv = MPI_Wait(&(bd_var_.req_recv[nb.buf_id]), &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI_Wait failed");

    // redundant; guaranteed with MPI_Wait
    bd_var_.flag[nb.buf_id] = BoundaryStatus::arrived;
    bytes_rcvd_ += bd_var_.recvbufsz[nb.buf_id];
  }

  // XXX: we explicitly add sends in receive boundary buffers
  // instead of the clearboundary thing (todo: remove at clearboundary)
  for (auto nb : pmb->nbrvec_snd_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    int rv = MPI_Wait(&(bd_var_.req_send[nb.buf_id]), &status);
    MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI_Wait failed");

    bd_var_.flag[nb.buf_id] = BoundaryStatus::arrived;
  }
}

void BoundaryVariable::DestroyBoundaryData(BoundaryData<>& bd) {
  MLOGIFR0(MLOG_DBG2, "Destroying boundary data");

  for (int n = 0; n < bd.kMaxNeighbor; n++) {
    if (bd.req_send[n] != MPI_REQUEST_NULL) MPI_Request_free(&bd.req_send[n]);
    if (bd.req_recv[n] != MPI_REQUEST_NULL) MPI_Request_free(&bd.req_recv[n]);
  }
}
}  // namespace topo
