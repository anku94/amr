//
// Created by Ankush J on 4/27/23.
//

#pragma once

#include <cassert>
#include <climits>
#include <string>

#include "common.h"
#include "reader_base.h"

#define FAILIO_IFLT(x, y)                                                     \
  if (x < y) {                                                                \
    logv(__LOG_ARGS__, LOG_ERRO, "Error reading file: %s\n", fpath_.c_str()); \
    return -1;                                                                \
  }

#define SAFE_READ(x)                 \
  n = fread(x, sizeof(int), 1, fd_); \
  FAILIO_IFLT(n, 1)

namespace amr {
class PeekableReader : public ReaderBase {
 protected:
  explicit PeekableReader(std::string fpath)
      : ReaderBase(std::move(fpath)),
        next_ts_(0),
        next_sub_ts_(0),
        peeked_(false) {}

  // Reader implementing this must move the fd cursor forward to next ts
  int ConsumePeek(int& ts, int& sub_ts) {
    if (!peeked_) {
      return PeakNext(ts, sub_ts);
    } else {
      ts = next_ts_;
      sub_ts = next_sub_ts_;
    }

    peeked_ = false;
    return feof(fd_) ? 0 : 1;
  }

  int PeakNext(int& ts, int& sub_ts) {
    ts = INT_MAX;
    sub_ts = INT_MAX;

    if (peeked_) {
      ts = next_ts_;
      sub_ts = next_sub_ts_;
      return feof(fd_) ? 0 : 1;
    }

    size_t n;

    n = fread(&next_ts_, sizeof(int), 1, fd_);
    if (n == 0 && feof(fd_)) {
      logv(__LOG_ARGS__, LOG_DBUG, "Reached end of file");
      return 0;
    } else {
      FAILIO_IFLT(n, 1);
    }

    SAFE_READ(&next_sub_ts_);
    peeked_ = true;

    ts = next_ts_;
    sub_ts = next_sub_ts_;

    return 1;
  }

 private:
  int next_ts_;
  int next_sub_ts_;
  bool peeked_;
};

class RefinementReader : public PeekableReader {
 public:
  explicit RefinementReader(std::string fpath)
      : PeekableReader(fpath + "/refinements.bin") {}

  int ReadTimestep(int& ts, int sub_ts, std::vector<int>& refs,
                   std::vector<int>& derefs) {
    ts = -1;

    int next_ts, next_sub_ts;

    // 0: no read, no more to come
    int rv = PeakNext(next_ts, next_sub_ts);
    if (rv == 0) return 0;

    if (sub_ts == next_sub_ts) {
      ts = next_ts;
      rv = ReadTimestepInternal(ts, sub_ts, refs, derefs);
      // return can't fail since we already peeked
      assert(rv == 1);
    } else if (sub_ts < next_sub_ts) {
      ts = -1;
      refs.resize(0);
      derefs.resize(0);
    } else {
      // read request for a future sub_ts, error
      logv(__LOG_ARGS__, LOG_ERRO, "Unexpected state in RefinementReader!!");
      ABORT("Unexpected state in RefinementReader");
    }

    return 1;
  }

 private:
  int ReadTimestepInternal(int ts, int sub_ts, std::vector<int>& refs,
                           std::vector<int>& derefs) {
    size_t n;

    int nrefs, nderefs;

    int rv = ConsumePeek(ts, sub_ts);
    if (rv == 0) return 0;

    SAFE_READ(&nrefs);
    refs.resize(nrefs);

    for (int i = 0; i < nrefs; i++) {
      SAFE_READ(&refs[i]);
    }

    SAFE_READ(&nderefs);
    derefs.resize(nderefs);

    for (int i = 0; i < nderefs; i++) {
      SAFE_READ(&derefs[i]);
    }

    return 1;
  }
};

class AssignmentReader : public PeekableReader {
 public:
  explicit AssignmentReader(std::string fpath)
      : PeekableReader(fpath + "/assignments.bin") {}

  int ReadTimestep(int& ts, int sub_ts, std::vector<int>& blocks) {
    ts = -1;

    int next_ts, next_sub_ts;

    // 0: no read, no more to come
    int rv = PeakNext(next_ts, next_sub_ts);
    if (rv == 0) return 0;

    if (sub_ts == next_sub_ts) {
      ts = next_ts;
      rv = ReadTimestepInternal(ts, sub_ts, blocks);
      // return can't fail since we already peeked
      assert(rv == 1);
    } else if (sub_ts < next_sub_ts) {
      // Should not happen with assignment reader, we expect
      // every sub_ts to have an entry
      logv(__LOG_ARGS__, LOG_WARN, "UNEXPECTED: sub_ts < next_sub_ts");
      blocks.resize(0);
    } else {
      // read request for a future sub_ts, error
      logv(__LOG_ARGS__, LOG_ERRO, "Unexpected state in RefinementReader!!");
      ABORT("Unexpected state in RefinementReader");
    }

    return 1;
  }

 private:
  int ReadTimestepInternal(int ts, int sub_ts, std::vector<int>& blocks) {
    int n;
    int nblocks;

    int rv = ConsumePeek(ts, sub_ts);
    if (rv == 0) return 0;

    SAFE_READ(&nblocks);
    blocks.resize(nblocks);

    for (int i = 0; i < nblocks; i++) {
      SAFE_READ(&blocks[i]);
    }

    return 1;
  }
};
}  // namespace amr

#undef FAILIO_IFLT
#undef SAFE_READ
