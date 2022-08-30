//
// Created by Ankush J on 8/29/22.
//

#include "trace_reader.h"

Status TraceReader::Read() {
  Status s = Status::OK;

  if (trace_file_ == "" or file_read_) {
    return s;
  }

  /* XXX: catch here is that Read can return FAIL the first time
   * and success on successive reads
   */
  file_read_ = true;

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
    s = ParseLine(buf, ret);
    if (s != Status::OK) return s;
  }

  fclose(f);

  PrintSummary();
  return Status::OK;
}

Status TraceReader::ParseLine(char* buf, size_t buf_sz) {
  Status s = Status::OK;

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

  /* XXX: parameterize phase_name maybe */
  if (strncmp(phase_name, "BoundaryComm", 12)) {
    return s;
  }

  logf(LOG_DBG2, "TS %d: Msg (%d -> %d), MsgSz: %dB", ts, rank, peer, msg_sz);

  if (s_or_r == 0) {
    ts_snd_map_[ts].push_back(std::pair<int, int>(peer, msg_sz));
  } else {
    ts_rcv_map_[ts].push_back(std::pair<int, int>(peer, msg_sz));
  }

  max_ts_ = std::max(ts, max_ts_);
  return s;
}
