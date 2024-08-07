//
// Created by Ankush J on 4/11/22.
//

#include "block.h"
#include "globals.h"

void BoundaryVariable::InitBoundaryData(BoundaryData<>& bd) {
  for (int n = 0; n < bd.kMaxNeighbor; n++) {
    bd.flag[n] = BoundaryStatus::waiting;
    bd.sflag[n] = BoundaryStatus::waiting;
    bd.req_send[n] = MPI_REQUEST_NULL;
    bd.req_recv[n] = MPI_REQUEST_NULL;
  }
}
void BoundaryVariable::SetupPersistentMPI() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    logv(__LOG_ARGS__, LOG_DBG2, "[SEND] Us: %d, Neighbor: %d (bufid: %d)", Globals::my_rank,
         nb.peer_rank, nb.buf_id);

    if (bd_var_.req_send[nb.buf_id] != MPI_REQUEST_NULL)
      MPI_Request_free(&bd_var_.req_send[nb.buf_id]);

    // buffer, msgsize, datatype, dest, tag, comm, req
    MPI_Send_init(bd_var_.sendbuf[nb.buf_id], nb.msg_sz, MPI_CHAR, nb.peer_rank, 0,
                  MPI_COMM_WORLD, &(bd_var_.req_send[nb.buf_id]));
    bd_var_.sendbufsz[nb.buf_id] = nb.msg_sz;
  }

  for (auto nb : pmb->nbrvec_rcv_) {
    logv(__LOG_ARGS__, LOG_DBG2, "[RECV] Us: %d, Neighbor: %d (bufid: %d)", Globals::my_rank,
         nb.peer_rank, nb.buf_id);

    if (bd_var_.req_recv[nb.buf_id] != MPI_REQUEST_NULL)
      MPI_Request_free(&bd_var_.req_recv[nb.buf_id]);

    MPI_Recv_init(bd_var_.recvbuf[nb.buf_id], nb.msg_sz, MPI_CHAR, nb.peer_rank, 0,
                  MPI_COMM_WORLD, &(bd_var_.req_recv[nb.buf_id]));
    bd_var_.recvbufsz[nb.buf_id] = nb.msg_sz;
  }
}

void BoundaryVariable::StartReceiving() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_rcv_) {
    int status = MPI_Start(&(bd_var_.req_recv[nb.buf_id]));
    if (status != MPI_SUCCESS) {
      logv(__LOG_ARGS__, LOG_ERRO, "MPI Start Failed");
    }

    logv(__LOG_ARGS__, LOG_DBG2, "Rank %d - Receive POSTED %d", Globals::my_rank, nb.buf_id);
  }
}
void BoundaryVariable::ClearBoundary() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_snd_) {
    bd_var_.flag[nb.buf_id] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.buf_id] = BoundaryStatus::waiting;

    // XXX: some sort of fence
    int status = MPI_Wait(&(bd_var_.req_send[nb.buf_id]), MPI_STATUS_IGNORE);
    MPI_CHECK(status, "MPI Wait Failed");

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
    logv(__LOG_ARGS__, LOG_DBG2, "Rank %d - Send START %d", Globals::my_rank, nb.buf_id);

    int status = MPI_Start(&(bd_var_.req_send[nb.buf_id]));
    MPI_CHECK(status, "MPI Start Failed");

    logv(__LOG_ARGS__, LOG_DBG2, "Rank %d - Send POSTED %d", Globals::my_rank, nb.buf_id);
  }
}
bool BoundaryVariable::ReceiveBoundaryBuffers() {
  bool bflag = true;
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  for (auto nb : pmb->nbrvec_rcv_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    int test;

    int status = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                            MPI_STATUS_IGNORE);
    MPI_CHECK(status, "MPI Iprobe Failed");

    status = MPI_Test(&(bd_var_.req_recv[nb.buf_id]), &test, MPI_STATUS_IGNORE);
    MPI_CHECK(status, "MPI Test Failed");

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

  for (auto nb : pmb->nbrvec_rcv_) {
    if (bd_var_.flag[nb.buf_id] == BoundaryStatus::arrived) continue;
    int status = MPI_Wait(&(bd_var_.req_recv[nb.buf_id]), MPI_STATUS_IGNORE);
    MPI_CHECK(status, "MPI_Wait failed");

    // redundant; guaranteed with MPI_Wait
    bd_var_.flag[nb.buf_id] = BoundaryStatus::arrived;
    bytes_rcvd_ += bd_var_.recvbufsz[nb.buf_id];
  }
}

void BoundaryVariable::DestroyBoundaryData(BoundaryData<>& bd) {
  for (int n = 0; n < bd.kMaxNeighbor; n++) {
    if (bd.req_send[n] != MPI_REQUEST_NULL) MPI_Request_free(&bd.req_send[n]);
    if (bd.req_recv[n] != MPI_REQUEST_NULL) MPI_Request_free(&bd.req_recv[n]);
  }
}
