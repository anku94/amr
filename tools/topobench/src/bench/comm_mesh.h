#pragma once

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
    MLOGIFR0(MLOG_INFO, "Resetting bvars and blocks");

    for (const auto& b : blocks_) {
      Status s = b->DestroyBoundaryData();
      if (s != Status::OK) return s;
    }

    blocks_.clear();

    return Status::OK;
  }

  Status DoCommunicationRound() {
    logger_.LogBegin();

    MLOGIFR0(MLOG_DBG0, "Start Receiving...");
    for (const auto& b : blocks_) {
      b->StartReceiving();
    }

    MLOGIFR0(MLOG_DBG0, "Sending Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->SendBoundaryBuffers();
    }

    MLOGIFR0(MLOG_DBG0, "Receiving Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->ReceiveBoundaryBuffers();
    }

    MLOGIFR0(MLOG_DBG0, "Receiving Remaining Boundary Buffers...");
    for (const auto& b : blocks_) {
      b->ReceiveBoundaryBuffersWithWait();
    }

    MLOGIFR0(MLOG_DBG0, "Clearing Boundaries...");
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
  friend class PlacementUtils;
};
}  // namespace topo::bench
