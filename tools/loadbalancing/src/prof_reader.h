#include "prof_base.h"

#include <common.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace amr {
class CSVProfileReader : public ProfileReader {
 public:
  explicit CSVProfileReader(const char* prof_csv_path,
                            ProfTimeCombinePolicy combine_policy)
      : ProfileReader(prof_csv_path, combine_policy),
        ts_(-1),
        eof_(false),
        prev_ts_(-1),
        prev_sub_ts_(-1),
        prev_bid_(-1),
        prev_time_(-1),
        first_read_(true),
        prev_set_(false) {
    Reset();
  }

  CSVProfileReader(const CSVProfileReader& other) = delete;

  int ReadNextTimestep(std::vector<int>& times) override {
    if (eof_) return -1;

    int nlines_read = 0;
    int nblocks = 0;

    while (nlines_read == 0) {
      nblocks = ReadTimestep(ts_, times, nlines_read);
      ts_++;
    }

    logf(LOG_DBUG, "[ProfReader] Timestep: %d, lines read: %d", ts_,
         nlines_read);

    return nblocks;
  }

  void Reset() {
    logf(LOG_DBG2, "[ProfReader] Reset: %s", csv_path_.c_str());
    SafeCloseFile();

    fd_ = fopen(csv_path_.c_str(), "r");

    if (fd_ == nullptr) {
      logf(LOG_ERRO, "[ProfReader] Unable to open: %s", csv_path_.c_str());
      ABORT("Unable to open specified CSV");
    }

    ts_ = -1;
    eof_ = false;
    prev_ts_ = prev_bid_ = prev_time_ = -1;
  }

 private:
  void ReadLine(char* buf, int max_sz) const {
    char* ret = fgets(buf, max_sz, fd_);
    int nbread = strlen(ret);
    if (ret[nbread - 1] != '\n') {
      ABORT("buffer too small for line");
    }

    logf(LOG_DBG2, "Line read: %s", buf);
  }

  void ReadHeader() {
    char header[1024];
    ReadLine(header, 1024);
  }

 public:
  /* Caller must zero the vector if needed!!
   * Returns: Number of blocks in current ts
   * (assuming contiguous bid allocation)
   */
  int ReadTimestep(int ts_to_read, std::vector<int>& times,
                   int& nlines_read) override {
    if (eof_) return -1;

    // Initialization hack
    if (fd_ == nullptr) Reset();

    if (first_read_) {
      ReadHeader();
      first_read_ = false;
    }

    int ts, sub_ts, rank, bid, time_us;
    sub_ts = ts_to_read;

    std::vector<int> times_cur;

    int max_bid = 0;

    if (prev_set_) {
      prev_set_ = false;
      if (prev_sub_ts_ == ts_to_read) {
        LogTime(times_cur, prev_bid_, prev_time_);
        nlines_read++;

        max_bid = std::max(max_bid, prev_bid_);
      } else if (prev_sub_ts_ < ts_to_read) {
        logf(LOG_WARN, "Somehow skipped ts %d data. Dropping...", prev_sub_ts_);
      } else if (prev_sub_ts_ > ts_to_read) {
        // Wait for ts to catch up
        return max_bid;
      }
    }

    while (true) {
      int nread;
      if ((nread = fscanf(fd_, "%d,%d,%d,%d,%d", &ts, &sub_ts, &rank, &bid,
                          &time_us)) == EOF) {
        eof_ = true;
        break;
      }

      if (sub_ts > ts_to_read) {
        prev_ts_ = ts;
        prev_sub_ts_ = sub_ts;
        prev_bid_ = bid;
        prev_time_ = time_us;
        prev_set_ = true;
        break;
      }

      max_bid = std::max(max_bid, bid);
      LogTime(times, bid, time_us);
      nlines_read++;
    }

    logf(LOG_DBG2, "[ProfReader] Times (cur): %s",
         SerializeVector(times_cur, 10).c_str());
    AddVec2Vec(times, times_cur);
    logf(LOG_DBG2, "[ProfReader] Times (tot): %s",
         SerializeVector(times_cur, 10).c_str());

    return max_bid + 1;
  }

  void LogTime(std::vector<int>& times, int bid, int time_us) const {
    if (times.size() <= bid) {
      times.resize(bid + 1, 0);
    }

    if (combine_policy_ == ProfTimeCombinePolicy::kUseFirst) {
      if (times[bid] == 0) times[bid] = time_us;
    } else if (combine_policy_ == ProfTimeCombinePolicy::kUseLast) {
      times[bid] = time_us;
    } else if (combine_policy_ == ProfTimeCombinePolicy::kAdd) {
      times[bid] += time_us;
    }
  }

  static void AddVec2Vec(std::vector<int>& dest, std::vector<int> const& src) {
    if (dest.size() < src.size()) {
      dest.resize(src.size(), 0);
    }

    for (int i = 0; i < src.size(); i++) {
      dest[i] += src[i];
    }
  }

  int ts_;
  bool eof_;

  int prev_ts_;
  int prev_sub_ts_;
  int prev_bid_;
  int prev_time_;

  bool first_read_;
  bool prev_set_;
};
}  // namespace amr
