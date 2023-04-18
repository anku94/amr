//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include "common.h"

#include <map>
#include <string>
#include <vector>

typedef std::pair<int, int> RankSizePair;

class TraceReader {
 public:
  TraceReader(const char* trace_root, int rank)
      : max_ts_(-1), file_read_(false) {
    if (trace_root == nullptr or trace_root[0] == '\0') {
      trace_file_ = "";
      return;
    }

    size_t buf_sz = trace_file_.size() + 64;
    char buf[buf_sz];
    snprintf(buf, buf_sz, "%s/msgs.%d.csv", trace_root, rank);
    trace_file_ = buf;
  }

  TraceReader(const char* trace_file) : trace_file_(trace_file), max_ts_(0) {}

  Status Read();

  int GetNumTimesteps() const { return max_ts_ + 1; }

  std::vector<RankSizePair> GetMsgsSent(int ts) { return ts_snd_map_[ts]; }

  std::vector<RankSizePair> GetMsgsRcvd(int ts) { return ts_rcv_map_[ts]; }

 private:
  Status ParseLine(char* buf, size_t buf_sz);

  void PrintSummary() {
    logf(LOG_INFO, "Timesteps upto ts %d discovered", max_ts_);
    for (size_t t = 0; t <= max_ts_; t++) {
      auto msgs_ts = ts_snd_map_[t];
      logf(LOG_DBUG, "[Send] TS %d: %zu msgs", t, msgs_ts.size());
    }

    for (size_t t = 0; t <= max_ts_; t++) {
      auto msgs_ts = ts_rcv_map_[t];
      logf(LOG_DBUG, "[Recv] TS %d: %zu msgs", t, msgs_ts.size());
    }
  }

  std::string trace_file_;
  std::map<int, std::vector<RankSizePair>> ts_snd_map_;
  std::map<int, std::vector<RankSizePair>> ts_rcv_map_;
  int max_ts_;
  bool file_read_;
};
