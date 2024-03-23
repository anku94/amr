#include "amr_util.h"

#include "../../tools/common/common.h"

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

void EnsureDirOrDie(const char* dir_path, int rank, int sub_rank) {
  if (sub_rank != 0) {
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
  int sub_id = rank / 64;
  int sub_rank = rank % 64;

  char subdir_path[4096];
  // snprintf(subdir_path, 4096, "%s/%s.%d", dir_path, fprefix, sub_id);
  // EnsureDirOrDie(subdir_path, rank, sub_rank);
  snprintf(subdir_path, 4096, "%s/%s", dir_path, fprefix);
  EnsureDirOrDie(subdir_path, rank, rank);

  // sleep(10);

  char fpath[8192];
  snprintf(fpath, 8192, "%s/%s.%d.%s", subdir_path, fprefix, rank, fmt);

  // logf(LOG_INFO, "Trying to create: %s", fpath);

  int attempts_rem = 4;
  int sleep_timer = 2;

  while((attempts_rem--) && (*file == nullptr)) {
    sleep(sleep_timer);
    sleep_timer *= 2;
    *file = fopen(fpath, "w+");
    if (*file == nullptr) {
      logf(LOG_INFO, "Rank: %d, error: %s", rank, strerror(errno));
    }
  }

  if (*file == nullptr) {
    ABORT("Failed to open CSV");
  }
}

};  // namespace tau
