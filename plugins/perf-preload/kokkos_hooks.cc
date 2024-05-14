#include "amr_monitor.h"
#include "common.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

extern "C" {
void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                          const uint32_t devInfoCount, void* deviceInfo) {
  if (amr::monitor == nullptr) {
    amr::Error(
        __LOG_ARGS__,
        "AMRMonitor not initialized by MPI_Init. This is unexpected behavior.");
  }
}

void kokkosp_finalize_library() {}

void kokkosp_begin_parallel_for(const char* name, uint32_t devid,
                                uint64_t* uniqueid) {}

void kokkosp_end_parallel_for(const uint64_t uniqueid) {}

void kokkosp_begin_parallel_scan(const char* name, uint32_t devid,
                                 uint64_t* uniqueid) {}

void kokkosp_end_parallel_scan(const uint64_t uniqueid) {}

void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid,
                                   uint64_t* uniqueid) {}

void kokkosp_end_parallel_reduce(const uint64_t uniqueid) {}

void kokkosp_push_profile_region(const char* name) {
  amr::monitor->LogStackBegin("profreg", name);
}

void kokkosp_pop_profile_region() { amr::monitor->LogStackEnd("profreg"); }
}  // extern "C"
