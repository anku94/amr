#!/usr/bin/env bash

REG_OSU=/users/ankushj/repos/intel-mpi-benchmarks/osumb-ompispack/osu-micro-benchmarks-5.8
PSM_OSU=/users/ankushj/repos/intel-mpi-benchmarks/osumb-mvapich2/osu-micro-benchmarks-5.8

. ~/spack/share/spack/setup-env.sh
spack load mpi

BIN=mpi/pt2pt/osu_bw

mpirun --mca btl tcp,self,vader --mca btl_tcp_if_include ibs2 --mca pml ob1 --host h0,h1 $REG_OSU/$BIN
LD_LIBRARY_PATH=/usr/lib64 /users/ankushj/repos/tau/mvapich2-install/bin/mpirun --host h0,h1 $PSM_OSU/$BIN
