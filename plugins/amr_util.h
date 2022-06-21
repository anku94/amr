#pragma once

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

};  // namespace tau
