//
// Created by Ankush J on 4/12/22.
//

#include "logger.h"

#include "block.h"

#include <inttypes.h>

void Logger::LogData(std::vector<std::shared_ptr<MeshBlock>>& blocks_) {
  total_sent_ = 0;
  total_rcvd_ = 0;

  for (auto b : blocks_) {
    total_sent_ += b->BytesSent();
    total_rcvd_ += b->BytesRcvd();
  }

  auto delta = end_ - start_;
  total_time_ += delta.count();
}

void Logger::Aggregate() {
  uint64_t global_sent, global_rcvd;
  double global_time;

  MPI_Reduce(&total_sent_, &global_sent, 1, MPI_UINT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_rcvd_, &global_rcvd, 1, MPI_UINT64_T, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_time_, &global_time, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  const uint64_t bytes_per_mb = 1ull << 30;
  double global_sent_mb = global_sent * 1.0 / bytes_per_mb;
  double global_rcvd_mb = global_rcvd * 1.0 / bytes_per_mb;

  if (Globals::my_rank == 0) {
    double sent_mbps = global_sent_mb / global_time;
    double rcvd_mbps = global_rcvd_mb / global_time;
    logf(LOG_INFO, "Bytes Exchanged: %" PRIu64 " B/%" PRIu64 " MB",
         global_sent, global_rcvd);
    logf(LOG_INFO, "Bytes Exchanged: %.2lf MB/%.2lf MB", global_sent_mb,
         global_rcvd_mb);
    logf(LOG_INFO, "Effective b/w SEND: %.2lf MB/s RECV: %.2lf MB/s", sent_mbps,
         rcvd_mbps);
  }
}
