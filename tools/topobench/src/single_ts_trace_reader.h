//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include "common.h"

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
    logvat0(__LOG_ARGS__, LOG_INFO, "Send count: %zu, Recv count: %zu",
            send_map_.size(), recv_map_.size());
  }

  std::string trace_file_;
  std::vector<CommNeighbor> send_map_;
  std::vector<CommNeighbor> recv_map_;
  bool file_read_;
};
