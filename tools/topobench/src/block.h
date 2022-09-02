//
// Created by Ankush J on 4/8/22.
//

#pragma once

#include "bvar.h"
#include "common.h"
#include "logger.h"

#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#define MPI_CHECK(status, msg) \
  if (status != MPI_SUCCESS) { \
    logf(LOG_ERRO, msg);       \
  }

struct NeighborBlock {
  int block_id;
  int peer_rank;
  int buf_id;
  int msg_sz;
};

class MeshBlock : public std::enable_shared_from_this<MeshBlock> {
 public:
  MeshBlock(int block_id) : block_id_(block_id) {}
  MeshBlock(const MeshBlock& other)
      : block_id_(other.block_id_),
        nbrvec_snd_(other.nbrvec_snd_),
        nbrvec_rcv_(other.nbrvec_rcv_) {}

  Status AddNeighborSendRecv(int block_id, int peer_rank, int msg_sz) {
    Status s;
    s = AddNeighborSend(block_id, peer_rank, msg_sz);
    if (s != Status::OK) return s;
    s = AddNeighborRecv(block_id, peer_rank, msg_sz);
    return s;
  }

  Status AddNeighborSend(int block_id, int peer_rank, int msg_sz) {
    int buf_id = nbrvec_snd_.size() + nbrvec_rcv_.size();
    nbrvec_snd_.push_back({block_id, peer_rank, buf_id, msg_sz});
    return Status::OK;
  }

  Status AddNeighborRecv(int block_id, int peer_rank, int msg_sz) {
    int buf_id = nbrvec_snd_.size() + nbrvec_rcv_.size();
    nbrvec_rcv_.push_back({block_id, peer_rank, buf_id, msg_sz});
    return Status::OK;
  }

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

    logf(LOG_DBUG, "Rank %d, Block ID %d, Neighbors: %s", Globals::my_rank,
         block_id_, nbrstr.c_str());
  }

  Status AllocateBoundaryVariables() {
    logf(LOG_DBUG, "Allocating boundary variables");
    pbval_ = std::make_unique<BoundaryVariable>(shared_from_this());
    pbval_->SetupPersistentMPI();
    logf(LOG_DBUG, "Allocating boundary variables - DONE!");
    return Status::OK;
  }

  Status DestroyBoundaryData() {
    pbval_.release();
    return Status::OK;
  }

  void StartReceiving() { pbval_->StartReceiving(); }

  void SendBoundaryBuffers() { pbval_->SendBoundaryBuffers(); }

  void ReceiveBoundaryBuffers() { pbval_->ReceiveBoundaryBuffers(); }

  void ReceiveBoundaryBuffersWithWait() {
    pbval_->ReceiveBoundaryBuffersWithWait();
  }

  void ClearBoundary() { pbval_->ClearBoundary(); }

  uint64_t BytesSent() const { return pbval_->bytes_sent_; }

  uint64_t BytesRcvd() const { return pbval_->bytes_rcvd_; }

 private:
  friend class BoundaryVariable;
  int block_id_;
  std::unique_ptr<BoundaryVariable> pbval_;
  std::vector<NeighborBlock> nbrvec_snd_;
  std::vector<NeighborBlock> nbrvec_rcv_;
};

class Mesh {
 public:
  Status AllocateBoundaryVariables() {
    for (const auto& b : blocks_) {
      Status s = b->AllocateBoundaryVariables();
      if (s != Status::OK) return s;
    }

    return Status::OK;
  }

  Status Reset() {
    for (const auto& b : blocks_) {
      Status s = b->DestroyBoundaryData();
      if (s != Status::OK) return s;
    }

    blocks_.clear();

    return Status::OK;
  }

  Status DoCommunicationRound() {
    logger_.LogBegin();

    logf(LOG_DBUG, "Start Receiving...");
    for (const auto& b : blocks_) {
      b->StartReceiving();
    }

    logf(LOG_DBUG, "Sending Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->SendBoundaryBuffers();
    }

    logf(LOG_DBUG, "Receiving Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->ReceiveBoundaryBuffers();
    }

    logf(LOG_DBUG, "Receiving Remaining Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->ReceiveBoundaryBuffersWithWait();
    }

    logf(LOG_DBUG, "Clearing Boundaries...");
    for (auto b : blocks_) {
      b->ClearBoundary();
    }

    logger_.LogEnd();
    logger_.LogData(blocks_);

    return Status::OK;
  }

  void PrintStats() {
    logger_.Aggregate();
  }

  void PrintConfig() {
    for (const auto& block : blocks_) {
      block->Print();
    }
  }

 private:
  Status AddBlock(const std::shared_ptr<MeshBlock>& block) {
    blocks_.push_back(block);
    return Status::OK;
  }

  std::vector<std::shared_ptr<MeshBlock>> blocks_;
  Logger logger_;

  friend class Topology;
};
