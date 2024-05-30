//
// Created by Ankush J on 8/29/22.
//

#include "trace_reader.h"

Status TraceReader::Read(int rank) {
  Status s = Status::OK;

  logvat0(__LOG_ARGS__, LOG_INFO, "[TraceReader] Reading trace file: %s",
       trace_file_.c_str());

  if (trace_file_ == "") {
    logv(__LOG_ARGS__, LOG_ERRO, "[TraceReader] No trace file provided");
    return Status::Error;
  }

  if (file_read_) {
    return Status::OK;
  }

  /* XXX: catch here is that Read can return FAIL the first time
   * and success on successive reads
   */
  file_read_ = true;

  logv(__LOG_ARGS__, LOG_DBUG, "[TraceReader] Reading %s\n",
       trace_file_.c_str());

  FILE *f = fopen(trace_file_.c_str(), "r");
  if (f == nullptr) {
    logv(__LOG_ARGS__, LOG_ERRO, "[TraceReader] Read Failed: %s",
         strerror(errno));
    s = Status::Error;
    return s;
  }

  size_t buf_sz = 4096;
  char buf[buf_sz];

  /* scan header */
  int ret = fscanf(f, "%4095[^\n]\n", buf);
  while (!feof(f)) {
    ret = fscanf(f, "%4095[^\n]\n", buf);
    if (ret == EOF or ret == 0)
      break;

    s = ParseLine(buf, ret, rank);
    if (s != Status::OK)
      return s;
  }
  // while (ret != EOF) {
  //   ret = fscanf(f, "%4095[^\n]\n", buf);
  //   if (ret == EOF)
  //     break;
  //   s = ParseLine(buf, ret, rank);
  //   if (s != Status::OK)
  //     return s;
  // }

  fclose(f);

  PrintSummary();
  return Status::OK;
}

Status TraceReader::ParseLine(char *buf, size_t buf_sz, const int rank) {
  Status s = Status::OK;

  logv(__LOG_ARGS__, LOG_DBG3, "Parsing: %s", buf);

  /* ts,blk_id,blk_rank,nbr_id,nbr_rank,msgsz,isflx */

  const char *fmtstr = "%d,%d,%d,%d,%d,%d,%d";

  int ts, send_rank, send_blk, recv_rank, recv_blk, msg_sz, is_flx;

  int ret = sscanf(buf, fmtstr, &ts, &send_blk, &send_rank, &recv_blk,
                   &recv_rank, &msg_sz, &is_flx);

  if (ret != 7) {
    return Status::Error;
  }

  logv(__LOG_ARGS__, LOG_DBG3, "TS %d: Msg (%d -> %d), MsgSz: %dB", ts,
       send_rank, recv_rank, msg_sz);

  if (send_rank == rank && !is_flx) {
    ts_snd_map_[ts].push_back({recv_blk, recv_rank, msg_sz});
  } else if (recv_rank == rank && !is_flx) {
    ts_rcv_map_[ts].push_back({send_blk, send_rank, msg_sz});
  }

  max_ts_ = std::max(ts, max_ts_);
  return s;
}
