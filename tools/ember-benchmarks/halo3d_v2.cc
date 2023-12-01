//
// Created by Ankush J on 11/30/23.
//

#include "block_common.h"
#include "common.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
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

  void DoIteration() {
    int req_count = 0;
    DoFace(req_count);
    DoEdge(req_count);

    MPI_Waitall(req_count, request_, status_);
    AssertAllSuccess(status_, req_count);
  }

  uint64_t GetBytesExchanged() const { return bytes_total_; }

 private:
  void DoFace(int& req_count) {
    auto face_neighbors = nrg_.GetFaceNeighbors();
    int face_sizes[] = {1000, 1000, 2000, 2000, 4000, 4000};
    for (size_t dest_idx = 0; dest_idx < face_neighbors.size(); dest_idx++) {
      int dest = face_neighbors[dest_idx];
      if (dest == -1) {
        continue;
      }

      auto& buf = buffer_suite_.face_buffers_[dest_idx];
      auto msg_size = face_sizes[dest_idx];
      DoCommSingle(buf, dest, msg_size, req_count);
    }
  }

  void DoEdge(int& req_count) {
    auto edge_neighbors = nrg_.GetEdgeNeighbors();
    int edge_sizes[] = {1000, 1000, 1000, 1000, 2000, 2000,
                        2000, 2000, 4000, 4000, 4000, 4000};
    for (size_t dest_idx = 0; dest_idx < edge_neighbors.size(); dest_idx++) {
      int dest = edge_neighbors[dest_idx];
      if (dest == -1) {
        continue;
      }

      auto& buf = buffer_suite_.edge_buffers_[dest_idx];
      auto msg_size = edge_sizes[dest_idx];
      DoCommSingle(buf, dest, msg_size, req_count);
    }
  }

  void DoCommSingle(BufferPair& buffer_pair, int dest, int msg_size,
                    int& req_count) {
    MPI_Isend(buffer_pair.send_buffer_.data(), buffer_pair.send_buffer_.size(),
              MPI_DOUBLE, dest, msg_size, MPI_COMM_WORLD,
              &request_[req_count++]);
    MPI_Irecv(buffer_pair.recv_buffer_.data(), buffer_pair.recv_buffer_.size(),
              MPI_DOUBLE, dest, msg_size, MPI_COMM_WORLD,
              &request_[req_count++]);

    bytes_total_ += msg_size;
  }

  void LogMPIStatus(MPI_Status* all_status, int n) {
    if (rank_ != 0) return;

    int nsuccess = 0;
    for (int i = 0; i < n; i++) {
      if (all_status[i].MPI_ERROR == MPI_SUCCESS) {
        nsuccess++;
      }
    }

    printf("MPI_Status: %d/%d success\n", nsuccess, n);
  }

  void AssertAllSuccess(MPI_Status* all_status, int n) {
    for (int i = 0; i < n; i++) {
      if (all_status[i].MPI_ERROR != MPI_SUCCESS) {
        printf("MPI_Status: %d/%d success\n", i, n);
        exit(1);
      }
    }
  }

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

class Block {
 public:
  explicit Block(EmberOpts& opts)
      : data_grid_(opts.nx, opts.nx, opts.nz),
        proc_grid_(opts.pex, opts.pey, opts.pez),
        iters_(opts.iterations),
        nvars_(opts.vars),
        sleep_(opts.sleep),
        map_file_(opts.map) {}

  void Run() {
    int my_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    RunRank(my_rank);
  }

  void DoSleep() {
    struct timespec sleep_ts {
      0, sleep_
    };
    struct timespec remain_ts {
      0, 0
    };

    if (nanosleep(&sleep_ts, &remain_ts) == EINTR) {
      while (nanosleep(&remain_ts, &remain_ts) == EINTR)
        ;
    }
  }

  void RunRank(int rank) {
    Triplet my_pos = PositionUtils::GetPosition(rank, proc_grid_);
    Communicator comm(rank, my_pos, proc_grid_, data_grid_, nvars_);

    if (IsCenter(my_pos, proc_grid_)) {
      printf("Doing communication for %d iterations.\n", iters_);
    }

    auto beg_us = GetMicros();

    for (int i = 0; i < iters_; i++) {
      // if (i == 0) DoSleep();
      comm.DoIteration();
    }

    auto time_us = GetMicros() - beg_us;
    double time_sec = time_us / 1e6;
    double size_kb = comm.GetBytesExchanged() / 1024.0;

    MPI_Barrier(MPI_COMM_WORLD);

    if (IsCenter(my_pos, proc_grid_)) {
      printf("%20s %20s %20s\n", "Time", "KBXchng/Rank-Max", "MB/S/Rank");
      printf("%20.3f %20.3f %20.3f\n", time_sec, size_kb,
             (size_kb / 1024.0) / time_sec);
    }
  }

  static bool IsCenter(Triplet pos, Triplet bounds) {
    return pos.x == bounds.x / 2 and pos.y == bounds.y / 2 and
           pos.z == bounds.z / 2;
  }

  static uint64_t GetMicros() {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
  }

 private:
  Triplet data_grid_;
  Triplet proc_grid_;
  int iters_;
  int nvars_;
  int sleep_;
  std::string map_file_;
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  EmberOpts opts = EmberUtils::ParseOptions(argc, argv);

  if (rank == 0) {
    printf("Running halo3d_v2\n");
    printf("%s\n", opts.ToString().c_str());
  }

  Block block(opts);
  block.Run();

  MPI_Finalize();
  return 0;
}
