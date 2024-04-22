#pragma once

#include <pdlfs-common/env.h>

#include "common.h"

extern const char* policy_file;

class CSVReader {
 public:
  CSVReader(const char* filename, pdlfs::Env* env)
      : filename_(filename), env_(env), eof_(false), fh_(nullptr) {
    if (!env_->FileExists(filename_)) {
      ABORT("File does not exist");
    }

    if (!env_->NewSequentialFile(filename, &fh_).ok()) {
      ABORT("Failed to open file");
    }
  }

  ~CSVReader() {
    if (fh_) {
      delete fh_;
    }
  }

  std::vector<double> ReadOnce() {
    auto line = ReadLine(fh_);
    auto vec_times = ParseCSVLine(line);

    auto line2 = ReadLine(fh_);
    auto vec_ranks = ParseCSVLine(line);

    return vec_times;
  }

  std::vector<double> ParseCSVLine(std::string const& line) {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
      values.push_back(std::stod(token));
    }

    return values;
  }

  void PreviewVector(std::vector<double> const& vec, int n) {
    n = std::min(n, (int)vec.size());
    std::stringstream msg;
    for (int i = 0; i < n; i++) {
      msg << vec[i] << ", ";
    }
    msg << "...";

    logv(__LOG_ARGS__, LOG_INFO, "Preview (%zu): %s\n", vec.size(),
         msg.str().c_str());
  }

  std::string ReadLine(pdlfs::SequentialFile* fh) {
    std::string line;

    while (!eof_ or !line_buf_.empty()) {
      // if line_buf_ has a newline char:
      // - append it to line, until that newline char
      // - set line_buf_ to the rest of the line
      // else read 1024 bytes and add to line_buf_

      auto newline_pos = line_buf_.find('\n');
      if (newline_pos != std::string::npos) {
        line += line_buf_.substr(0, newline_pos);
        line_buf_ = line_buf_.substr(newline_pos + 1);
        break;
      }

      char buf[1024];
      pdlfs::Slice result;
      pdlfs::Status s = fh->Read(1024, &result, buf);
      if (!s.ok() or result.size() == 0) {
        eof_ = true;
        break;
      } else {
        line_buf_ += result.ToString();
      }
    }

    return line;
  }

 private:
  const char* filename_;
  pdlfs::Env* env_;
  std::string line_buf_;
  bool eof_;
  pdlfs::SequentialFile* fh_;
};
