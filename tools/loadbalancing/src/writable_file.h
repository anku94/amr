//
// Created by Ankush J on 6/30/23.
//

#pragma once

#include <pdlfs-common/env.h>
#include <pdlfs-common/status.h>

#define SAFE_IO(func, msg) \
  s = func;                    \
  if (!s.ok()) {               \
    ABORT(msg);                \
  }

//
// RAII-safe WritableFile with move semantics
//
namespace amr {
class WritableFile {
 public:
  WritableFile(pdlfs::Env* const env, const std::string& fpath)
      : env_(env) {
    if (env_->FileExists(fpath.c_str())) {
      logf(LOG_WARN, "Overwriting file: %s", fpath.c_str());
      env_->DeleteFile(fpath.c_str());
    }

    pdlfs::Status s;
    pdlfs::WritableFile* fd;
    SAFE_IO(env_->NewWritableFile(fpath.c_str(), &fd),
                "Unable to open WriteableFile!");

    fd_ptr_.reset(fd);
  }

  ~WritableFile() {
    if (fd_ptr_) {
      pdlfs::Status s;
      SAFE_IO(fd_ptr_->Close(), "Close failed");
    }
  }

  void Append(const std::string& data) {
    pdlfs::Status s;
    SAFE_IO(fd_ptr_->Append(data), "Append failed");
  }

  void Append(const char* data, int data_len) {
    pdlfs::Status s;
    SAFE_IO(fd_ptr_->Append(pdlfs::Slice(data, data_len)), "Append failed");
  }

 private:
  pdlfs::Env* const env_;
  std::unique_ptr<pdlfs::WritableFile> fd_ptr_;
};
}  // namespace amr
