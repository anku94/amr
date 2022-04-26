#pragma once

#define MATCHES(s) (strncmp(block_name, s, strlen(s)) == 0)

namespace amr {
enum class AmrFunc {
  RedistributeAndRefine,
  SendBoundBuf,
  RecvBoundBuf,
  SendFluxCor,
  RecvFluxCor,
  MakeOutputs,
  Unknown
};

AmrFunc ParseBlock(const char* block_name) {
  if (MATCHES("RedistributeAndRefineMeshBlocks")) {
    return AmrFunc::RedistributeAndRefine;
  } else if (MATCHES("Task_SendBoundaryBuffers_MeshData")) {
    return AmrFunc::SendBoundBuf;
  } else if (MATCHES("Task_ReceiveBoundaryBuffers_MeshData")) {
    return AmrFunc::RecvBoundBuf;
  } else if (MATCHES("Task_SendFluxCorrection")) {
    return AmrFunc::SendFluxCor;
  } else if (MATCHES("Task_ReceiveFluxCorrection")) {
    return AmrFunc::RecvFluxCor;
  } else if (MATCHES("MakeOutputs")) {
    return AmrFunc::MakeOutputs;
  }

  return AmrFunc::Unknown;
}
};  // namespace amr
