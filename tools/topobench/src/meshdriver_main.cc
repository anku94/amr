#include <mpi.h>
#include <cstdio>

#include "bench/mesh_driver.h"
#include "amr/mesh_utils.h"
#include "topo_common.h"

using namespace topo::amr;

int main(int argc, char *argv[]) {
    FLAGS_logtostderr = true;
    FLAGS_log_prefix = false;
    google::InitGoogleLogging(argv[0]);

    MPI_Init(&argc, &argv);

    int my_rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    MLOG(LOG_INFO, "Hello, world! I am rank %d of %d", my_rank, nranks);

    topo::bench::MeshDriverOpts opts;
    opts.mesh_dims = Vec3i(2, 2, 2);
    opts.max_reflvl = 4;
    topo::bench::MeshDriver driver(opts);
    driver.PrintOpts();

    MPI_Finalize();
    return 0;
}