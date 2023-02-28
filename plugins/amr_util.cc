#include "amr_util.h"

#include "../tools/common.h"

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MATCHES(s) (strncmp(block_name, s, strlen(s)) == 0)

namespace tau {

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

void EnsureDirOrDie(const char* dir_path, int rank) {
  if (rank != 0) {
    return;
  }

  if (mkdir(dir_path, S_IRWXU)) {
    if (errno != EEXIST) {
      logf(LOG_ERRO, "Unable to create directory: %s", dir_path);
      ABORT("Unable to create directory");
    }
  }
}

void EnsureFileOrDie(FILE** file, const char* dir_path, const char* fprefix,
                     const char* fmt, int rank) {
  char subdir_path[4096];
  snprintf(subdir_path, 4096, "%s/%s", dir_path, fprefix);
  EnsureDirOrDie(subdir_path, rank);

  char fpath[4096];
  snprintf(fpath, 4096, "%s/%s/%s.%d.%s", dir_path, fprefix, fprefix, rank, fmt);

  int attempts_rem = 3;
  int sleep_timer = 1;

  while((attempts_rem--) && (*file == nullptr)) {
    sleep(sleep_timer);
    sleep_timer *= 2;
    *file = fopen(fpath, "w+");
  }

  if (*file == nullptr) {
    ABORT("Failed to open CSV");
  }
}

};  // namespace tau
