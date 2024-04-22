//
// Created by Ankush J on 4/28/23.
//

#pragma once

#include <utility>

#include "common.h"

namespace amr {
class ReaderBase {
 public:
  explicit ReaderBase(std::string fpath)
      : fpath_(std::move(fpath)), fd_(nullptr) {
    fd_ = fopen(fpath_.c_str(), "rb");
    if (fd_ == nullptr) {
      logv(__LOG_ARGS__, LOG_ERRO, "Unable to open file: %s\n", fpath_.c_str());
      ABORT("Unable to open file");
    }
  }

  ReaderBase(const ReaderBase& other) = delete;

  ReaderBase& operator=(const ReaderBase&& other) = delete;

  ReaderBase(ReaderBase&& other) noexcept
      : fpath_(other.fpath_), fd_(other.fd_) {
    if (this != &other) {
      other.fd_ = nullptr;
    }
  }

  ~ReaderBase() {
    if (fd_ != nullptr) {
      fclose(fd_);
      fd_ = nullptr;
    }
  }

 protected:
  std::string const fpath_;
  FILE* fd_;
};
}  // namespace amr
