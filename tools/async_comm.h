//
// Created by Ankush J on 4/8/22.
//
#pragma once

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ABORT(msg)                           \
  {                                          \
    fprintf(stderr, msg "\n");               \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
  }

double cur_time() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  double tsec = t.tv_sec + 1e-9 * t.tv_nsec;
  return tsec;
}

class MPIComm {
 public:
  MPIComm() : size_(0), my_rank_(0), xfer_mb_total_(0), xfer_time_total_(0) {
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    if (size_ != 2) ABORT("Need n=2");

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);

    // printf("Rank %d\n", my_rank_);
  }

  void SendInit(int drank) {
    for (size_t i = 0; i < kMaxReqCnt; i++) {
      MPI_Send_init(bufs_[i], kMsgSize, MPI_CHAR, drank, 0, MPI_COMM_WORLD,
                    &reqs_[i]);
    }
  }

  void ActuallySend(int drank) {
    for (size_t i = 0; i < kMaxReqCnt; i++) {
      MPI_Start(&reqs_[i]);
    }
  }

  void WaitAll() {
    for (size_t i = 0; i < kMaxReqCnt; i++) {
      MPI_Wait(&reqs_[i], MPI_STATUS_IGNORE);
      // printf("Rank %d sent req_id %zu\n", my_rank_, i);
    }
  }

  void SendOnce(int drank) {
    double time_send = cur_time();

    ActuallySend(drank);

    double time_wait = cur_time();

    WaitAll();

    double time_end = cur_time();

    // printf("\nTime Taken:\n"
    // "\tInit Send: \t%.3lf\n"
    // "\tActual Send: \t%.3lf\n"
    // "\tWait Time: \t%.3lf\n",
    // time_send - time_start, time_wait - time_send, time_end - time_wait);

    double xfer_data_mb = kMsgSize * kMaxReqCnt * 1e-6;
    // double xfer_time = time_wait - time_send;
    double xfer_time = time_end - time_send;

    xfer_mb_total_ += xfer_data_mb;
    xfer_time_total_ += xfer_time;

    // printf("Total Reqs: %zu, %zuB each\n", kMaxReqCnt, kMsgSize);
  }

  void ReceiveOnce(int srank) {
    double recv_start = cur_time();

    for (size_t i = 0; i < kMaxReqCnt; i++) {
      MPI_Recv(bufs_[i], kMsgSize, MPI_CHAR, srank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    double recv_end = cur_time();

    double xfer_data_mb = kMsgSize * kMaxReqCnt * 1e-6;
    double xfer_time = recv_end - recv_start;

    xfer_mb_total_ += xfer_data_mb;
    xfer_time_total_ += xfer_time;

    // printf("\nRecv b/w: %.2lfMB/s\n", xfer_data_mb / xfer_time);
  }

  void Send(int drank) { SendOnce(drank); }

  void Receive(int srank) { ReceiveOnce(srank); }

  void Run() {
    if (my_rank_ == 0) {
      SendInit(1);
      for (int i = 0; i < 100; i++) Send(1);

    } else {
      for (int i = 0; i < 100; i++) Receive(0);
    }

    printf("Effective b/w: %.2lfMB/s\n", xfer_mb_total_ / xfer_time_total_);
    // printf("Total Data Xferred: %.1lfMB\n", xfer_mb_total_);
  }

 private:
  static const size_t kMsgSize = 4096;
  static const size_t kMaxReqCnt = 1 << 14;

  int size_;
  int my_rank_;

  char bufs_[kMaxReqCnt][kMsgSize];
  MPI_Request reqs_[kMaxReqCnt];

  double xfer_mb_total_;
  double xfer_time_total_;
};