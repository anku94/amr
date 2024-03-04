#include <mpi.h>
#include <stdio.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

extern "C" {

// PMPI wrapper for MPI_Barrier
int MPI_Barrier(MPI_Comm comm) {
    static int (*MPI_Barrier_real)(MPI_Comm comm) = NULL;
    if (!MPI_Barrier_real) {
        MPI_Barrier_real = (int (*)(MPI_Comm))dlsym(RTLD_NEXT, "MPI_Barrier");
    }

    printf("Intercepting MPI_Barrier\n");

    // Call the actual MPI_Barrier function
    return MPI_Barrier_real(comm);
}

// Kokkos Tools interface function for starting a region
void kokkosp_push_region(const char* regionName) {
    printf("Starting Kokkos region: %s\n", regionName);
}

// Kokkos Tools interface function for ending a region
void kokkosp_pop_region() {
    printf("Ending Kokkos region\n");
}

} // extern "C"
