#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

static int count = 0;

extern "C" {

// Profiling tool registration
void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                          const uint32_t devInfoCount, void* deviceInfo) {
  std::cout << "[ " << count++ << " ] "
    << "KokkosP: Example Library Initialized (sequence: " << loadSeq
            << ")" << std::endl;
}

void kokkosp_finalize_library() {
  std::cout << "[ " << count++ << " ] "
    << "KokkosP: Example Library Finalized" << std::endl;
}

void kokkosp_begin_parallel_for(const char* name, uint32_t devID,
                                uint64_t* kernID) {}

void kokkosp_end_parallel_for(uint64_t kernID) {}

void kokkosp_begin_parallel_scan(const char* name, uint32_t devID,
                                 uint64_t* kernID) {}

void kokkosp_end_parallel_scan(uint64_t kernID) {}

void kokkosp_begin_parallel_reduce(const char* name, uint32_t devID,
                                   uint64_t* kernID) {}

void kokkosp_end_parallel_reduce(uint64_t kernID) {}

void kokkosp_push_profile_region(const char* name) {
  std::cout << "[ " << count++ << " ] "
    << "Entering Kokkos region: " << name << std::endl;
}

void kokkosp_pop_profile_region(const char* name) {
  std::cout << "[ " << count++ << " ] "
    << "Exiting Kokkos region: " << name << std::endl;
}



// PMPI wrapper for MPI_Barrier
int MPI_Barrier(MPI_Comm comm) {
  static int (*MPI_Barrier_real)(MPI_Comm comm) = NULL;
  if (!MPI_Barrier_real) {
    MPI_Barrier_real = (int (*)(MPI_Comm))dlsym(RTLD_NEXT, "MPI_Barrier");
  }

  std::cout << "[ " << count++ << " ] " << "Entering MPI_Barrier" << std::endl;

  // Call the actual MPI_Barrier function
  return MPI_Barrier_real(comm);
}

}  // extern "C"
