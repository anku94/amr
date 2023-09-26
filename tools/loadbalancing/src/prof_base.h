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

  const std::string csv_path_;
  ProfTimeCombinePolicy combine_policy_;
  FILE* fd_;
};

class BinProfileReader : public ProfileReader {
 public:
  BinProfileReader(const char* prof_csv_path,
                   ProfTimeCombinePolicy combine_policy)
      : ProfileReader(prof_csv_path, combine_policy),
        eof_(false),
        nts_(-1) {}

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
      logf(LOG_ERRO, "[ProfReader] Invalid ts_to_read: %d", ts_to_read);
      ABORT("Invalid ts_to_read.");
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

    times.resize(nblocks);
    nitems = fread(times.data(), sizeof(int), nblocks, fd_);
    ASSERT_NREAD(nitems, nblocks);

    if (ts_to_read != ts_read) {
      logf(LOG_WARN, "Expected: %d, Read: %d", ts_to_read, ts_read);
    }

    nlines_read = nblocks;
    return nblocks;
  }

  int ReadNextTimestep(std::vector<int>& times) override {
    if (eof_) return -1;
    ReadHeader();

    int ts_read;
    int nblocks;

    int nitems = fread(&ts_read, sizeof(int), 1, fd_);
    if (nitems < 1) {
      eof_ = true;
      return -1;
    }

    nitems = fread(&nblocks, sizeof(int), 1, fd_);
    ASSERT_NREAD(nitems, 1);

    times.resize(nblocks);
    nitems = fread(times.data(), sizeof(int), nblocks, fd_);
    ASSERT_NREAD(nitems, nblocks);

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
