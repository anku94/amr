//
// Created by Ankush J on 4/8/22.
//

#pragma once

#include "common.h"
#include <mpi.h>
#include <vector>

#define NMAX_NEIGHBORS 56

struct Globals {
  int my_rank;
};

enum class BoundaryStatus { waiting,
                            arrived,
                            completed };

template<int n = NMAX_NEIGHBORS>
struct BoundaryData {
  static constexpr int kMaxNeighbor = n;
  int nbmax;
  BoundaryStatus flag[kMaxNeighbor], sflag[kMaxNeighbor];
  std::vector<char *> buffers;
  MPI_Request req_send[kMaxNeighbor], req_recv[kMaxNeighbor];
};

class BoundaryVariable {
 public:
  void SetupPersistentMPI();
  void SendBoundaryBuffers();
  void ReceiveBoundaryBuffers();

 private:
  BoundaryData<> bd_var_, bd_var_flcor_;
};

struct NeighborBlock {
  int block_id;
  int peer_rank;
};

class MeshBlock {
 public:
  MeshBlock(int block_id) : block_id_(block_id) {}
  MeshBlock(const MeshBlock &other) : block_id_(other.block_id_),
                                      nbrvec_(other.nbrvec_) {}
  Status AddNeighbor(int block_id, int peer_rank) {
    nbrvec_.push_back({block_id, peer_rank});
    return Status::OK;
  }

 private:
  int block_id_;
  std::unique_ptr<BoundaryVariable> pbval_;
  std::vector<NeighborBlock> nbrvec_;
};

class Mesh {
 public:
  Status AddBlock(const MeshBlock &block) {
    blocks_.push_back(block);
    return Status::OK;
  }

 private:
  std::vector<MeshBlock> blocks_;
};