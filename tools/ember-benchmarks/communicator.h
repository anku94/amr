#pragma once

#include "block_common.h"

#include <vector>
#include <mpi.h>

class BufferPair {
 public:
  BufferPair(int size) : send_buffer_(size, 0), recv_buffer_(size, 0) {
    for (int i = 0; i < size; i++) {
      send_buffer_[i] = i;
    }
  }
  std::vector<double> send_buffer_;
  std::vector<double> recv_buffer_;
};

class BufferSuite {
 public:
  BufferSuite(Triplet n, int nvars)
      : n_(n),
        nvars_(nvars),
        face_buffers_{
            {n.y * n.z * nvars}, {n.y * n.z * nvars}, {n.x * n.z * nvars},
            {n.x * n.z * nvars}, {n.x * n.y * nvars}, {n.x * n.y * nvars},
        },
        edge_buffers_{
            {n.z * nvars}, {n.x * nvars}, {n.z * nvars}, {n.x * nvars},
            {n.y * nvars}, {n.y * nvars}, {n.y * nvars}, {n.y * nvars},
            {n.z * nvars}, {n.x * nvars}, {n.z * nvars}, {n.x * nvars},
        } {}

  Triplet n_;
  const int nvars_;

  BufferPair face_buffers_[6];
  BufferPair edge_buffers_[12];
};

class Communicator {
 public:
  Communicator(int rank, Triplet my_pos, Triplet bounds, Triplet n, int nvars)
      : rank_(rank),
        my_pos_(my_pos),
        bounds_(bounds),
        nrg_(my_pos_, bounds_),
        buffer_suite_(n, nvars),
        bytes_total_(0) {}

  void DoIteration();

  void DoPoll(int num_requests);

  uint64_t GetBytesExchanged() const { return bytes_total_; }

 private:
  void DoFace(int& req_count);

  void DoEdge(int& req_count);

  void DoCommSingle(BufferPair& buffer_pair, int dest, int msg_tag,
                    int& req_count);

  void LogMPIStatus(MPI_Status* all_status, int n);

  void AssertAllSuccess(MPI_Status* all_status, int n);

 private:
  int rank_;
  Triplet my_pos_;
  Triplet bounds_;
  NeighborRankGenerator nrg_;
  BufferSuite buffer_suite_;
  uint64_t bytes_total_;

  MPI_Status status_[52]{};
  MPI_Request request_[52]{};
};
