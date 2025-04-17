#pragma once

#include "amr/block.h"
#include "logger.h"
#include "topo_types.h"

namespace topo::bench {
class CommMesh {
 public:
  Status AllocateBvars() {
    for (const auto& b : blocks_) {
      Status s = b->AllocateBoundaryVariables();
      if (s != Status::OK) return s;
    }

    return Status::OK;
  }

  Status ResetBvarsAndBlocks() {
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

  void PrintStats() { logger_.Aggregate(); }

  void PrintConfig() {
    for (const auto& block : blocks_) {
      block->Print();
    }
  }

 private:
  Status AddBlock(const MeshBlockRef& block) {
    blocks_.push_back(block);
    return Status::OK;
  }

  std::vector<MeshBlockRef> blocks_;
  Logger logger_;

  friend class MeshGenerator;
};
}  // namespace topo::bench
