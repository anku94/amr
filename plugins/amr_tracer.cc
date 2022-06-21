#include "amr_tracer.h"
#include "tau_types.h"

namespace tau {

void AMRTracer::ProcessTriggerMsg(void *data) {
  int rank; int size;
  int global_min, global_max;
  int global_sum; float sum_, avg_, min_, max_;

  int local = *((int*)(data));

  PMPI_Reduce(&local, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&local, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&local, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
 
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    sum_ = global_sum;
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    fprintf(stderr, "Avg, min, max are %f %d %d \n", (sum_/size), global_min, global_max);
  }

  PMPI_Bcast(&local, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return;
}
}
