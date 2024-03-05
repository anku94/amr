#include "amr_monitor.h"
#include "common.h"

#include <cstdio>
#include <mpi.h>
#include <pdlfs-common/env.h>

extern "C" {
int MPI_Init(int* argc, char*** argv) {
  int rv = PMPI_Init(argc, argv);

  if (rv != MPI_SUCCESS) {
    return rv;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  amr::monitor = std::make_unique<amr::AMRMonitor>(pdlfs::Env::Default(), rank);

  return rv;
}

int MPI_Finalize() {
  amr::monitor.reset();

  int rv = PMPI_Finalize();
  return rv;
}

//
// Begin MPI Collective Hooks. These are all the same template.
// This has not been validated yet to be exhaustive.
//

int MPI_Barrier(MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Barrier(comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Barrier", ts_end - ts_beg);
  return rv;
}

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Bcast(buffer, count, datatype, root, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Bcast", ts_end - ts_beg);
  return rv;
}

int MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
               void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                       recvtype, root, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Gather", ts_end - ts_beg);
  return rv;
}

int MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, const int* recvcounts, const int* displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                        displs, recvtype, root, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Gatherv", ts_end - ts_beg);
  return rv;
}

int MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                        recvtype, root, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Scatter", ts_end - ts_beg);
  return rv;
}

int MPI_Scatterv(const void* sendbuf, const int* sendcounts, const int* displs,
                 MPI_Datatype sendtype, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype, int root, MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf,
                         recvcount, recvtype, root, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Scatterv", ts_end - ts_beg);
  return rv;
}

int MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Allgather", ts_end - ts_beg);
  return rv;
}

int MPI_Allgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                   void* recvbuf, const int* recvcounts, const int* displs,
                   MPI_Datatype recvtype, MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                           displs, recvtype, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Allgatherv", ts_end - ts_beg);
  return rv;
}

int MPI_Alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                         recvtype, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Alltoall", ts_end - ts_beg);
  return rv;
}

int MPI_Alltoallv(const void* sendbuf, const int* sendcounts,
                  const int* sdispls, MPI_Datatype sendtype, void* recvbuf,
                  const int* recvcounts, const int* rdispls,
                  MPI_Datatype recvtype, MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf,
                          recvcounts, rdispls, recvtype, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Alltoallv", ts_end - ts_beg);
  return rv;
}

int MPI_Reduce(const void* sendbuf, void* recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Reduce", ts_end - ts_beg);
  return rv;
}

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Allreduce", ts_end - ts_beg);
  return rv;
}

int MPI_Reduce_scatter(const void* sendbuf, void* recvbuf,
                       const int* recvcounts, MPI_Datatype datatype, MPI_Op op,
                       MPI_Comm comm) {
  auto ts_beg = amr::monitor->Now();
  int rv =
      PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
  auto ts_end = amr::monitor->Now();

  amr::monitor->LogMPICollective("MPI_Reduce_scatter", ts_end - ts_beg);
  return rv;
}

//
// END MPI Collective Hooks
//
}  // extern "C"
