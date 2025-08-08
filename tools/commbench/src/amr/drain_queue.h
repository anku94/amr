#pragma once

#include <mpi.h>

#include <list>
#include <memory>

#include "amr_types.h"
#include "tools-common/globals.h"
#include "tools-common/logging.h"

#define MPI_CHECK_STATUS(rv, mpi_status, msg) \
  if (rv != MPI_SUCCESS) {                    \
    CheckMpiStatus(rv, mpi_status, msg);      \
  }

inline void CheckMpiStatus(int rv, int mpi_status, const char* msg) {
  if (rv == MPI_SUCCESS) return;

  char errbuf[MPI_MAX_ERROR_STRING];
  int errlen;
  MPI_Error_string(mpi_status, errbuf, &errlen);
  MLOG(MLOG_ERRO, "%s: %s", msg, errbuf);
  MPI_Abort(MPI_COMM_WORLD, mpi_status);
}

namespace topo {
struct SendDrainQueueElement {
  MpiReqUref request;
  int ntests;  // Number of MPI_Tests

  SendDrainQueueElement(MpiReqUref request, int ntests)
      : request(std::move(request)), ntests(ntests) {}
};

class SendDrainQueue {
  // Collect std::shared_ptr<MPI_Request> objects

 public:
  void AddSendRequest(MpiReqUref request) {
    queue_.emplace_back(std::move(request), 0);
  };

  void TryDraining() {
    for (auto it = queue_.begin(); it != queue_.end();) {
      int flag;
      MPI_Status status;
      int rv = MPI_Test(it->request.get(), &flag, &status);
      MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI_Test failed!");

      if (flag) {
        it = queue_.erase(it);
      } else {
        it->ntests++;
        if (it->ntests > 100) {
          MLOGIFR0(MLOG_INFO, "[DrainQueue] MPI_Test failed for %d tries!",
                   it->ntests);
          continue;
        }

        it++;
      }
    }
  };

  void ForceDrain() {
    if (queue_.size() == 0) {
      return;
    }

    MLOGIFR0(MLOG_INFO, "[DrainQueue] Forcing drain of %d requests at rank %d",
             queue_.size(), Globals::my_rank);

    for (auto it = queue_.begin(); it != queue_.end();) {
      MPI_Status status;
      // MLOGIFR0(MLOG_INFO, "Not calling MPI_Wait() because PSM is a moron!");
      int rv = MPI_Wait(it->request.get(), &status);
      MPI_CHECK_STATUS(rv, status.MPI_ERROR, "MPI_Wait failed!");
      it++;
    }
  };

  ~SendDrainQueue() {
    if (queue_.size() > 0) {
      MLOGIFR0(MLOG_INFO, "[DrainQueue] %d requests at rank %d", queue_.size(),
               Globals::my_rank);
      ForceDrain();
    }
  };

 private:
  std::list<SendDrainQueueElement> queue_;
};
}  // namespace topo