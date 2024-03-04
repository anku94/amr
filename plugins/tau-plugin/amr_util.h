#pragma once

#include <stdio.h>

namespace tau {

enum class AmrFunc {
  RedistributeAndRefine,
  SendBoundBuf,
  RecvBoundBuf,
  SendFluxCor,
  RecvFluxCor,
  MakeOutputs,
  Unknown
};

AmrFunc ParseBlock(const char* block_name);

void EnsureDirOrDie(const char* dir_path, int rank);

void EnsureFileOrDie(FILE** file, const char* dir_path, const char* fprefix,
                     const char* fmt, int rank);

};  // namespace tau
