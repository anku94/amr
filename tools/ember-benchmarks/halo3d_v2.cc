//
// Created by Ankush J on 11/30/23.
//

#include "block_common.h"
#include "common_flags.h"

#include <cerrno>
#include <mpi.h>
#include <unistd.h>

#include "communicator.h"

std::unique_ptr<RankMap> RANK_MAP;

class Block {
public:
  explicit Block(EmberOpts &opts)
      : data_grid_(opts.nx, opts.nx, opts.nz),
        proc_grid_(opts.pex, opts.pey, opts.pez), iters_(opts.iterations),
        nvars_(opts.vars), sleep_(opts.sleep) {}

  void Run() {
    int my_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int log_rank = RANK_MAP->GetLogicalRank(my_rank);
    if (my_rank == 0) {
      printf("Rank %d is mapped to %d\n", my_rank, log_rank);
    }

    RunRank(log_rank);
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
};

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  EmberOpts opts = EmberUtils::ParseOptions(argc, argv);
  RANK_MAP = std::make_unique<RankMap>(nranks, opts.map);

  if (rank == 0) {
    printf("Running halo3d_v2\n");
    printf("%s\n", opts.ToString().c_str());
  }

  Block block(opts);
  block.Run();

  MPI_Finalize();
  return 0;
}
