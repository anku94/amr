#pragma once

#include "amr/block.h"
#include "logger.h"
#include "topo_types.h"

namespace topo {
class CommMesh {
 public:
  // CommMesh: constructor
  CommMesh() = default;

  // AllocateBvars: allocate boundary variables for all blocks
  Status AllocateBvars() {
    for (const auto& b : blocks_) {
      Status s = b->AllocateBoundaryVariables();
      if (s != Status::OK) return s;
    }

    return Status::OK;
  }

  // ResetBvarsAndBlocks: reset boundary variables and blocks
  Status ResetBvarsAndBlocks() {
    MLOGIFR0(MLOG_INFO, "Resetting bvars and blocks");

    for (const auto& b : blocks_) {
      Status s = b->DestroyBoundaryData();
      if (s != Status::OK) return s;
    }

    blocks_.clear();

    return Status::OK;
  }

  // DoCommunicationRound: do and log a comm round
  Status DoCommunicationRound() {
    // logger_.LogBegin();

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

    // logger_.LogEnd();
    // logger_.DrainBlockData(blocks_);

    return Status::OK;
  }

  // GenerateStats: Gather all stats using collectives, and print/log them
  // Creates one row in the log csv
  void GenerateStats(ExtraMetricVec& extra_metrics, const char* log_fpath) {
    // logger_.AggregateAndWrite(extra_metrics, log_fpath);
  }

  // PrintConfig: ??
  void PrintConfig() {
    for (const auto& block : blocks_) {
      block->Print();
    }
  }

 private:
  // AddBlock: Add a preconfigured block to the list
  Status AddBlock(const MeshBlockRef& block) {
    blocks_.push_back(block);
    return Status::OK;
  }

  int nblocks_global_ = 0;  // total block count, only for logger
 public:
  std::vector<MeshBlockRef> blocks_;  // mesh blocks
 private:
  Logger logger_;  // data logger

  friend class MeshGenerator;
  friend class PlacementUtils;
};
}  // namespace topo
