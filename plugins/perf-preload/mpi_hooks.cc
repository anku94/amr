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
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int nranks;
  PMPI_Comm_size(MPI_COMM_WORLD, &nranks);

  amr::monitor = std::make_unique<amr::AMRMonitor>(pdlfs::Env::Default(), rank, nranks);

  amr::Info(__LOG_ARGS__, "AMRMonitor initialized on rank %d", rank);

  return rv;
}

int MPI_Finalize() {
  amr::monitor.reset();

  PMPI_Barrier(MPI_COMM_WORLD);

  int rv = PMPI_Finalize();
  return rv;
}

//
// MPI Point-to-Point Hooks.
//

int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm) {
  int rv = PMPI_Send(buf, count, datatype, dest, tag, comm);
  amr::monitor->LogMPISend(dest, datatype, count);
  return rv;
}

int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request* request) {
  int rv = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
  amr::monitor->LogMPISend(dest, datatype, count);
  return rv;
}

int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status* status) {
  int rv = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  amr::monitor->LogMPIRecv(source, datatype, count);
  return rv;
}

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request* request) {
  int rv = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  amr::monitor->LogMPIRecv(source, datatype, count);
  return rv;
}

//
// BEGIN MPI Collective Hooks. These are all the same template.
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

//
// END MPI Collective Hooks
//
}  // extern "C"
