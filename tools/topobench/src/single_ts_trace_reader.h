//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include "common.h"
#include "globals.h"

#include <string>
#include <vector>

typedef std::pair<int, int> RankSizePair;

class SingleTimestepTraceReader {
public:
  SingleTimestepTraceReader(const char *trace_file)
      : trace_file_(trace_file), file_read_(false) {}

  Status Read(int rank);

  std::vector<CommNeighbor> GetMsgsSent() { return send_map_; }

  std::vector<CommNeighbor> GetMsgsRcvd() { return recv_map_; }

private:
  Status ParseLine(char *buf, size_t buf_sz, const int rank);

  void PrintSummary() {
    if (Globals ::my_rank == 0) {
      logv(::pdlfs ::Logger ::Default(),
           "/users/ankushj/repos/parthenon-vibe/amr-umbrella/build/"
           "amr-tools-prefix/src/amr-tools/tools/topobench/src/"
           "single_ts_trace_reader.h",
           29, 3, "Send count: %zu, Recv count: %zu", send_map_.size(),
           recv_map_.size());
    } else {
      logv(::pdlfs ::Logger ::Default(),
           "/users/ankushj/repos/parthenon-vibe/amr-umbrella/build/"
           "amr-tools-prefix/src/amr-tools/tools/topobench/src/"
           "single_ts_trace_reader.h",
           29, 3 + 1, "Send count: %zu, Recv count: %zu", send_map_.size(),
           recv_map_.size());
    };
  }

  std::string trace_file_;
  std::vector<CommNeighbor> send_map_;
  std::vector<CommNeighbor> recv_map_;
  bool file_read_;
};
