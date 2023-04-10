#include <common.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace amr {
class ProfileReader {
 public:
  ProfileReader(const char* prof_csv_path)
      : csv_path_(prof_csv_path),
        csv_fd_(nullptr),
        ts_(-1),
        eof_(false),
        prev_ts_(-1),
        prev_bid_(-1),
        prev_time_(-1) {
    Reset();
  }

  ProfileReader(const ProfileReader& other) = delete;

  /* Following two snippets are to support move semantics
   * for classes with non-RAII resources (in this case, FILE* ptrs)
   * Another way to do this is to wrap the FILE* in a RAII class
   * or a unique pointer
   */
  ProfileReader(ProfileReader&& rhs)
      : csv_path_(rhs.csv_path_),
        csv_fd_(rhs.csv_fd_),
        ts_(rhs.ts_),
        eof_(rhs.eof_),
        prev_ts_(rhs.prev_ts_),
        prev_bid_(rhs.prev_bid_),
        prev_time_(rhs.prev_time_) {
    if (this != &rhs) {
      rhs.csv_fd_ = nullptr;
    }
  }

  ProfileReader& operator=(ProfileReader&& rhs) {
    if (this != &rhs) {
      SafeCloseFile();
      csv_fd_ = rhs.csv_fd_;
      rhs.csv_fd_ = nullptr;
    }

    csv_path_ = rhs.csv_path_;
    ts_ = rhs.ts_;
    eof_ = rhs.eof_;
    prev_ts_ = rhs.prev_ts_;
    prev_bid_ = rhs.prev_bid_;
    prev_time_ = rhs.prev_time_;

    return *this;
  }

  ~ProfileReader() { SafeCloseFile(); }

  int ReadNextTimestep(std::vector<int>& times) {
    if (eof_) return -1;
    int nblocks = ReadTimestep(ts_ + 1, times);
    ts_++;

    return nblocks;
  }

  void Reset() {
    logf(LOG_DBG2, "Reset: %s", csv_path_);
    SafeCloseFile();

    // if (csv_fd_ != nullptr) {
    // printf("closing fd 2: %p\n", csv_fd_);
    // fclose(csv_fd_);
    // csv_fd_ = nullptr;
    // }

    csv_fd_ = fopen(csv_path_, "r");

    if (csv_fd_ == nullptr) {
      ABORT("Unable to open specified CSV");
    }

    ts_ = -1;
    eof_ = false;
    prev_ts_ = prev_bid_ = prev_time_ = -1;
  }

 private:
  void ReadLine(char* buf, int max_sz) {
    char* ret = fgets(buf, 1024, csv_fd_);
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

  /* Caller must zero the vector if needed!!
   * Returns: Number of blocks in current ts
   * (assuming contiguous bid allocation)
   */
  int ReadTimestep(int ts_to_read, std::vector<int>& times) {
    if (eof_) return -1;

    if (ts_to_read == 0) {
      ReadHeader();
    }

    int ts, bid, time_us;
    ts = ts_to_read;

    int max_bid = 0;

    if (prev_ts_ >= 0) {
      if (prev_ts_ == ts_to_read) {
        LogTime(times, prev_bid_, prev_time_);

        max_bid = std::max(max_bid, prev_bid_);
        prev_ts_ = prev_bid_ = prev_time_ = -1;
      } else if (prev_ts_ < ts_to_read) {
        logf(LOG_WARN, "Somehow skipped ts %d data. Dropping...", prev_ts_);
        prev_ts_ = prev_bid_ = prev_time_ = -1;
      }
    }

    while (true) {
      int nread;
      if ((nread = fscanf(csv_fd_, "%d,%d,%d", &ts, &bid, &time_us)) == EOF) {
        eof_ = true;
        break;
      }

      if (ts > ts_to_read) {
        prev_ts_ = ts;
        prev_bid_ = bid;
        prev_time_ = time_us;
        break;
      }

      max_bid = std::max(max_bid, bid);
      LogTime(times, bid, time_us);
    }

    LogVector(times);

    return max_bid + 1;
  }

  void LogTime(std::vector<int>& times, int bid, int time_us) {
    if (times.size() <= bid) {
      times.resize(bid + 1, 0);
    }

    times[bid] += time_us;
  }

  void LogVector(std::vector<int>& v) {
    std::stringstream ss;
    ss << "(" << v.size() << " items): ";
    for (auto n : v) {
      ss << n << " ";
    }

    logf(LOG_DBUG, "%s", ss.str().c_str());
  }

  void SafeCloseFile() {
    if (csv_fd_) {
      fclose(csv_fd_);
      csv_fd_ = nullptr;
    }
  }

  const char* csv_path_;
  FILE* csv_fd_;
  int ts_;
  bool eof_;

  int prev_ts_;
  int prev_bid_;
  int prev_time_;
};
}  // namespace amr
