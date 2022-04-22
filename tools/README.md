Topology Simulator
=================

Replicates boundary variable communication in Parthenon.

Build and run as:

`mpirun -n <num_ranks> ./topobench -b <num_blocks_per_rank> -r <num_rounds> -s <msg_size> -t [0|1]`

1. `-t=0` indicates `NeighborTopology::Ring`. Each rank has 2 neighbors, and all blocks within a rank have their
   neighbors in the same two physical ranks. This topology minimizes the number of actual neighbors for a rank.
2. `-t=1` indicates `NeighborTopology::AllToAll`. In this topology, blocks have their neighbors spread out among all
   ranks. The topology generation algorithm requires `<num_blocks_per_rank> > <num_ranks>` to hold to generate a
   topology that suits these constraints.

The total amount of communication undertaken by a rank remains the same in both the cases. Only the physical neighbor
links change.

Example invocations:

`mpirun -n 64 -b 64 -r 128 -s 4096 -t 0 # ring`

`mpirun -n 64 -b 64 -r 128 -s 4096 -t 1 # all-to-all`


Source Code
-----------

1. `Block::DoCommunication()` in `block.h` describes the MPI calls and the order they're invoked in, to begin a round of
   communication.
2. `topology.h` contains the two topology generation methods. It is not necessary to understand them - it can be assumed
   that they generate meshblocks with the necessary topologies.