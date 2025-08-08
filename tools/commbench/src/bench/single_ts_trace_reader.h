//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include <string>
#include <vector>

#include "tools-common/common.h"
#include "tools-common/globals.h"
#include "tools-common/logging.h"

typedef std::pair<int, int> RankSizePair;

namespace topo {
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
    MLOGIFR0(MLOG_INFO, "Send count: %zu, Recv count: %zu", send_map_.size(),
             recv_map_.size());
  }

  std::string trace_file_;
  std::vector<CommNeighbor> send_map_;
  std::vector<CommNeighbor> recv_map_;
  bool file_read_;
};
}  // namespace topo
