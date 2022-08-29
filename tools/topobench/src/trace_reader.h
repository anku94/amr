//
// Created by Ankush J on 8/29/22.
//

#pragma once

#include "common.h"

#include <map>
#include <string>
#include <vector>

class TraceReader {
 public:
  TraceReader(const char* trace_root, int rank) : max_ts_(0) {
    size_t buf_sz = trace_file_.size() + 64;
    char buf[buf_sz];
    snprintf(buf, buf_sz, "%s/msgs/msgs.%d.csv", trace_root, rank);
    trace_file_ = buf;
  }

  TraceReader(const char* trace_file) : trace_file_(trace_file), max_ts_(0) {}

  Status Read() {
    Status s;

    FILE* f = fopen(trace_file_.c_str(), "r");
    if (f == nullptr) {
      logf(LOG_ERRO, "[TraceReader] Read Failed: %s", strerror(errno));
      s = Status::Error;
      return s;
    }

    size_t buf_sz = 4096;
    char buf[buf_sz];

    /* scan header */
    int ret = fscanf(f, "%4095[^\n]\n", buf);

    while (ret != EOF) {
      ret = fscanf(f, "%4095[^\n]\n", buf);
      logf(LOG_INFO, "%s", buf);
      s = ParseLine(buf, ret);
      if (s != Status::OK) return s;
    }

    fclose(f);

    PrintSummary();
    return Status::OK;
  }

 private:
  Status ParseLine(char* buf, size_t buf_sz) {
    /* rank|peer|ts|phase_name|msg_id|sorr|msg_sz|timestamp */
    int rank, peer, ts, msg_id, s_or_r, msg_sz;
    char phase_name[128];
    long long unsigned int timestamp;

    const char* fmtstr =
        "%d|%d|%d|%127[^|]|"
        "%d|%d|%d|%llu";

    int ret = sscanf(buf, fmtstr, &rank, &peer, &ts, &phase_name, &msg_id,
                     &s_or_r, &msg_sz, &timestamp);

    if (ret != 8) {
      return Status::Error;
    }

    if (s_or_r == 0) {
      logf(LOG_DBG2, "TS %d: Msg (%d -> %d), MsgSz: %dB", ts, rank, peer,
           msg_sz);

      ts_msg_map_[ts].push_back(std::pair<int, int>(peer, msg_sz));
      max_ts_ = std::max(ts, max_ts_);
    }

    return Status::OK;
  }

  void PrintSummary() {
    logf(LOG_INFO, "Timesteps upto ts %d discovered", max_ts_);
    for (size_t t = 0; t <= max_ts_; t++) {
      auto msgs_ts = ts_msg_map_[t];
      logf(LOG_DBUG, "TS %d: %zu msgs", t, msgs_ts.size());
    }
  }

  std::string trace_file_;
  std::map<int, std::vector<std::pair<int, int>>> ts_msg_map_;
  int max_ts_;
};
