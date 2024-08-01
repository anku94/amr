#include <Kokkos_Core.hpp>
#include <iostream>
#include <mpi.h>

#include "perfsignal.h"

PerfManager perf;

void basic_communication(MPI_Comm comm, int my_rank, int nranks) {
  perf.Resume();

  int msg[2] = {my_rank, nranks};
  int recv_msg[2];
  MPI_Request send_request, recv_request;
  MPI_Status status;

  int partner_rank = (my_rank % 2 == 0) ? my_rank + 1 : my_rank - 1;
  if (partner_rank < 0 || partner_rank >= nranks) {
    // No valid partner
    return;
  }

  // Post a non-blocking receive
  MPI_Irecv(recv_msg, 2, MPI_INT, partner_rank, 0, comm, &recv_request);

  // Send the message non-blocking
  MPI_Isend(msg, 2, MPI_INT, partner_rank, 0, comm, &send_request);

  // Wait for both send and receive to complete
  MPI_Wait(&send_request, &status);
  MPI_Wait(&recv_request, &status);

  // Optional: Print the received message for debugging
  printf("Rank %d received from rank %d: my_rank=%d, nranks=%d\n", my_rank,
         partner_rank, recv_msg[0], recv_msg[1]);

  perf.Pause();
}

void SmallWorkRegion(double& num) {
  Kokkos::Profiling::pushRegion("SmallWorkRegion");

  for (int i = 0; i < 1000; i++) {
    num += (1.0 / num) * (1.0 / num);
  }

  Kokkos::Profiling::popRegion();
}

void BigWorkRegion(double& num) {
  Kokkos::Profiling::pushRegion("BigWorkRegion");

  for (int i = 0; i < 10000; i++) {
    num += (1.0 / num) * (1.0 / num);
  }

  Kokkos::Profiling::popRegion();
}

void Timestep(int rank, int size, double& num) {
  Kokkos::Profiling::pushRegion("Timestep");

  basic_communication(MPI_COMM_WORLD, rank, size);
  SmallWorkRegion(num);
  BigWorkRegion(num);

  Kokkos::Profiling::popRegion();
}

void Init(int rank, int size) {
  // Kokkos parallel region (replace this with actual computation)
  Kokkos::Profiling::pushRegion("HelloWorldRegion");

  // Print from each process
  std::cout << "Hello from MPI process " << rank << " out of " << size
            << std::endl;

  Kokkos::Profiling::popRegion();
}

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  Init(rank, size);
  double num = 1.0;
  for (int t = 0; t < 3; t++) {
    Timestep(rank, size, num);
  }

  if (rank == 0) {
    std::cout << "Final num (ignore): " << num << std::endl;
  }

  // Finalize Kokkos
  Kokkos::finalize();

  MPI_Barrier(MPI_COMM_WORLD);

  // Finalize MPI
  MPI_Finalize();

  return 0;
}
