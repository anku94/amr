#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    {
        // Kokkos parallel region (replace this with actual computation)
        Kokkos::Profiling::pushRegion("HelloWorldRegion");

        // Print from each process
        std::cout << "Hello from MPI process " << rank << " out of " << size << std::endl;

        Kokkos::Profiling::popRegion();
    }

    // Finalize Kokkos
    Kokkos::finalize();

    MPI_Barrier(MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
