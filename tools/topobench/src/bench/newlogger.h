#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>

#include "amr/block.h"
#include "metric_utils.h"

namespace topo {
struct RoundStats {
  int ts;                  // timestep
  int round;               // round within timestep
  int nblocks;             // number of blocks
  uint64_t totbytes_sent;  // total bytes sent
  uint64_t totbytes_rcvd;  // total bytes received
  uint64_t totcnt_sent;    // total count sent
  uint64_t totcnt_rcvd;    // total count received
  double totdur_msg_comm;  // dur of msg comm
  double totdur_ms_bar;    // dur at rank0 after bar
};

class NewLogger {
  using MeshBlockRef = std::shared_ptr<MeshBlock>;

 public:
  NewLogger() = default;

  void LogRoundBegin() { start_us_ = NowMicros(); }

  void LogCommEnd() { comm_end_us_ = NowMicros(); }

  void LogBarrierEnd(int ts, int round, int nblocks, std::vector<MeshBlockRef> &blocks) {
    auto barend_us = NowMicros();

    RoundStats rs;
    rs.ts = ts;
    rs.round = round;
    rs.nblocks = nblocks;
    rs.totbytes_sent = 0;
    rs.totbytes_rcvd = 0;
    rs.totcnt_sent = 0;
    rs.totcnt_rcvd = 0;
    rs.totdur_msg_comm = comm_end_us_ - start_us_;
    rs.totdur_ms_bar = barend_us - start_us_;

    for (auto &b : blocks) {
      rs.totbytes_sent += b->BytesSent();
      rs.totbytes_rcvd += b->BytesRcvd();
      rs.totcnt_sent += b->CountSent();
      rs.totcnt_rcvd += b->CountRcvd();
    }

    round_stats_.push_back(rs);
  }

  void AggregateAndWrite(ExtraMetricVec &extra_metrics, const char *log_fpath);

 private:
  uint64_t start_us_{0};                 // round start time (us)
  uint64_t comm_end_us_{0};              // comm end time (us)
  std::vector<RoundStats> round_stats_;  // stats for each round

  uint64_t NowMicros() const {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
  }
};
}  // namespace topo
