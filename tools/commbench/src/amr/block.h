//
// Created by Ankush J on 4/8/22.
//

#pragma once

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

#include "amr_types.h"
#include "bvar.h"
#include "mesh_utils.h"
#include "tools-common/common.h"
#include "tools-common/globals.h"
#include "tools-common/logging.h"
#include "topo_types.h"

#define MPI_CHECK(status, msg) \
  if (status != MPI_SUCCESS) { \
    MLOG(MLOG_ERRO, msg);      \
    ABORT(msg);                \
  }

namespace topo {

//
// MeshBlock: represents a single AMR block
// - Holds local mesh relationships using NeighborBlock
//
class MeshBlock : public std::enable_shared_from_this<MeshBlock> {
 public:
  MeshBlock(int block_id) : block_id_(block_id) {}
  MeshBlock(const MeshBlock &other)
      : block_id_(other.block_id_),
        nbrvec_snd_(other.nbrvec_snd_),
        nbrvec_rcv_(other.nbrvec_rcv_) {}

  // AddNeighborSendRecv: add a neighbor to send to and recv from
  // (Note that this currently takes two buf id's, even though we could share)
  Status AddNeighborSendRecv(int block_id, int peer_rank, int msg_sz, int tag) {
    Status s;
    s = AddNeighborSend(block_id, peer_rank, msg_sz, tag);
    if (s != Status::OK) return s;
    s = AddNeighborRecv(block_id, peer_rank, msg_sz, tag);
    return s;
  }

  // AddNeighborSend: add a neighbor to send to
  // For convenience we only use one of sendbuf and recvbuf from a buf_id
  Status AddNeighborSend(int block_id, int peer_rank, int msg_sz, int tag) {
    int buf_id = nbrvec_snd_.size() + nbrvec_rcv_.size();
    nbrvec_snd_.push_back({block_id, peer_rank, buf_id, msg_sz, tag});

    int total = nbrvec_snd_.size() + nbrvec_rcv_.size();

    if (total > BoundaryData<>::kMaxNeighbor) {
      MLOG(MLOG_ERRO, "Exceeded max neighbors");
      ABORT("Exceeded max neighbors");

      return Status::Error;
    }

    return Status::OK;
  }

  // AddNeighborRecv: add a neighbor to recv from
  // For convenience we only use one of sendbuf and recvbuf from a buf_id
  Status AddNeighborRecv(int block_id, int peer_rank, int msg_sz, int tag) {
    int buf_id = nbrvec_snd_.size() + nbrvec_rcv_.size();
    nbrvec_rcv_.push_back({block_id, peer_rank, buf_id, msg_sz, tag});

    int total = nbrvec_snd_.size() + nbrvec_rcv_.size();

    if (total > BoundaryData<>::kMaxNeighbor) {
      MLOG(MLOG_ERRO, "Exceeded max neighbors");
      ABORT("Exceeded max neighbors");

      return Status::Error;
    }

    return Status::OK;
  }

  // Print: print local mesh relationships
  void Print() {
    std::string nbrstr = "[Send] ";

    for (auto nbr : nbrvec_snd_) {
      nbrstr += std::to_string(nbr.block_id) + "/" +
                std::to_string(nbr.peer_rank) + ",";
    }

    nbrstr += ", [Recv] ";
    for (auto nbr : nbrvec_rcv_) {
      nbrstr += std::to_string(nbr.block_id) + "/" +
                std::to_string(nbr.peer_rank) + ",";
    }

    MLOG(MLOG_DBG0, "Rank %d, Block ID %d, Neighbors: %s", Globals::my_rank,
         block_id_, nbrstr.c_str());
  }

  // AllocateBoundaryVariables: allocate MPI requests
  Status AllocateBoundaryVariables() {
    MLOG(MLOG_DBG1, "Allocating boundary variables");
    pbval_ = std::make_unique<BoundaryVariable>(shared_from_this());
    pbval_->SetupPersistentMPI();
    MLOG(MLOG_DBG2, "Allocating boundary variables - DONE!");
    return Status::OK;
  }

  Status DestroyBoundaryData() {
    MLOGIFR0(MLOG_DBG1, "Destroying boundary data at bid %d", block_id_);
    pbval_.reset();
    return Status::OK;
  }

  // StartReceiving: wraps bvar StartReceiving
  void StartReceiving() { pbval_->StartReceiving(); }

  // SendBoundaryBuffers: wraps bvar SendBoundaryBuffers
  void SendBoundaryBuffers() { pbval_->SendBoundaryBuffers(); }

  // ReceiveBoundaryBuffers: wraps bvar ReceiveBoundaryBuffers
  void ReceiveBoundaryBuffers() { pbval_->ReceiveBoundaryBuffers(); }

  // ReceiveBoundaryBuffersWithWait: wraps bvar ReceiveBoundaryBuffersWithWait
  void ReceiveBoundaryBuffersWithWait() {
    pbval_->ReceiveBoundaryBuffersWithWait();
  }

  // ClearBoundary: wraps bvar ClearBoundary
  void ClearBoundary() { pbval_->ClearBoundary(); }

  // BytesSent: wraps bvar BytesSent
  uint64_t BytesSent() const { return pbval_->bytes_sent_; }

  // BytesRcvd: wraps bvar BytesRcvd
  uint64_t BytesRcvd() const { return pbval_->bytes_rcvd_; }

  // CountSent: wraps bvar CountSent
  uint64_t CountSent() const { return pbval_->sendcnt_; }

  // CountRcvd: wraps bvar CountRcvd
  uint64_t CountRcvd() const { return pbval_->recvcnt_; }

 private:
  friend class BoundaryVariable;
  int block_id_;                             // our block id [0, nblocks)
  std::unique_ptr<BoundaryVariable> pbval_;  // bvars (only one for now)
  std::vector<NeighborBlock> nbrvec_snd_;    // nbrs we send to
  std::vector<NeighborBlock> nbrvec_rcv_;    // nbrs we recv from
};
}  // namespace topo
