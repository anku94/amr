//
// Created by Ankush J on 9/25/23.
//

#pragma once

namespace amr {
class ProfileReader {
 public:
  explicit ProfileReader(const char* prof_csv_path,
                         ProfTimeCombinePolicy combine_policy)
      : csv_path_(prof_csv_path),
        combine_policy_(combine_policy),
        fd_(nullptr) {}

  ~ProfileReader() { SafeCloseFile(); }

  /* Following two snippets are to support move semantics
   * for classes with non-RAII resources (in this case, FILE* ptrs)
   * Another way to do this is to wrap the FILE* in a RAII class
   * or a unique pointer
   */

  ProfileReader& operator=(ProfileReader&& rhs) = delete;

  ProfileReader(ProfileReader&& rhs) noexcept
      : csv_path_(rhs.csv_path_),
        combine_policy_(rhs.combine_policy_),
        fd_(rhs.fd_) {
    if (this != &rhs) {
      rhs.fd_ = nullptr;
    }
  }

  virtual int ReadNextTimestep(std::vector<int>& times) = 0;

  virtual int ReadTimestep(int ts_to_read, std::vector<int>& times,
                           int& nlines_read) = 0;

 protected:
  void SafeCloseFile() {
    if (fd_) {
      fclose(fd_);
      fd_ = nullptr;
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

  const std::string csv_path_;
  ProfTimeCombinePolicy combine_policy_;
  FILE* fd_;
};

class BinProfileReader : public ProfileReader {
 public:
  BinProfileReader(const char* prof_csv_path,
                   ProfTimeCombinePolicy combine_policy)
      : ProfileReader(prof_csv_path, combine_policy), eof_(false), nts_(-1) {}

#define ASSERT_NREAD(a, b)                                                     \
  if (a != b) {                                                                \
    logf(LOG_ERRO, "[BinProfileReader] Read error. Expected: %d, read: %d", b, \
         a);                                                                   \
    ABORT("Read Error");                                                       \
  }
  int ReadTimestep(int ts_to_read, std::vector<int>& times,
                   int& nlines_read) override {
    if (eof_) return -1;
    ReadHeader();

    if (ts_to_read >= nts_) {
      eof_ = true;
      return -1;
    }

    if (ts_to_read < 0) {
      logf(LOG_WARN, "[ProfReader] Invalid ts_to_read: %d", ts_to_read);
      // ABORT("Invalid ts_to_read.");
      return 0;
    }

    int ts_read;
    int nblocks;

    int nitems = fread(&ts_read, sizeof(int), 1, fd_);
    if (nitems < 1) {
      eof_ = true;
      return -1;
    }

    nitems = fread(&nblocks, sizeof(int), 1, fd_);
    ASSERT_NREAD(nitems, 1);

    std::vector<int> times_cur(nblocks, 0);
    if (times.size() < nblocks) {
      times.resize(nblocks, 0);
    }

    nitems = fread(times_cur.data(), sizeof(int), nblocks, fd_);
    ASSERT_NREAD(nitems, nblocks);

    if (ts_to_read != ts_read) {
      logf(LOG_WARN, "Expected: %d, Read: %d", ts_to_read, ts_read);
    }

    logf(LOG_DBG2, "[ProfReader] Times (cur): %s",
         SerializeVector(times_cur, 10).c_str());
    AddVec2Vec(times, times_cur);
    logf(LOG_DBG2, "[ProfReader] Times (tot): %s",
         SerializeVector(times, 10).c_str());

    nlines_read = nblocks;
    return nblocks;
  }

  int ReadNextTimestep(std::vector<int>& times) override {
    if (eof_) return -1;
    ReadHeader();

    logf(LOG_WARN, "Not implemented.");

    // int ts_read;
    // int nblocks;

    // int nitems = fread(&ts_read, sizeof(int), 1, fd_);
    // if (nitems < 1) {
    // eof_ = true;
    // return -1;
    // }

    // nitems = fread(&nblocks, sizeof(int), 1, fd_);
    // ASSERT_NREAD(nitems, 1);

    // times.resize(nblocks);
    // nitems = fread(times.data(), sizeof(int), nblocks, fd_);
    // ASSERT_NREAD(nitems, nblocks);

    return 0;
  }

 private:
  void ReadHeader() {
    if (fd_) return;

    fd_ = fopen(csv_path_.c_str(), "r");

    if (fd_ == nullptr) {
      logf(LOG_ERRO, "[ProfReader] Unable to open: %s", csv_path_.c_str());
      ABORT("Unable to open specified BIN.");
    }

    eof_ = false;

    int nread = fread(&nts_, sizeof(int), 1, fd_);
    if (nread != 1) {
      logf(LOG_ERRO, "[ProfReader] Unable to read nts: %s", csv_path_.c_str());
      ABORT("Unable to read nts.");
    }
  }

  bool eof_;
  int nts_;
};
}  // namespace amr
