#pragma once

#include "block.h"

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

    logv(__LOG_ARGS__, LOG_DBUG, "Start Receiving...");
    for (const auto& b : blocks_) {
      b->StartReceiving();
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Sending Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->SendBoundaryBuffers();
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Receiving Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->ReceiveBoundaryBuffers();
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Receiving Remaining Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->ReceiveBoundaryBuffersWithWait();
    }

    logv(__LOG_ARGS__, LOG_DBUG, "Clearing Boundaries...");
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
