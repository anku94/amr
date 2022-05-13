#!/usr/bin/env bash

TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-mpich-2004
TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004

PLUGIN=amr
PLUGIN_OUT=$(echo $PLUGIN | sed 's/_/-/g')

g++ \
  -Wl,-Bsymbolic-functions \
  -Wl,-z,relro \
  -I/usr/include/x86_64-linux-gnu/mpich \
  -L/usr/lib/x86_64-linux-gnu \
  -lmpichcxx \
  -lmpich \
  -I$TAU_ROOT/tau-2.31/include \
  -DPROFILING_ON \
  -DTAU_GNU \
  -DTAU_DOT_H_LESS_HEADERS \
  -DTAU_MPI \
  -DTAU_UNIFY \
  -DTAU_MPI_THREADED \
  -DTAU_LINUX_TIMERS \
  -DTAU_MPIGREQUEST \
  -DTAU_MPIDATAREP \
  -DTAU_MPIERRHANDLER \
  -DTAU_MPICONSTCHAR \
  -DTAU_MPIATTRFUNCTION \
  -DTAU_MPITYPEEX \
  -DTAU_MPIADDERROR \
  -DTAU_LARGEFILE \
  -D_LARGEFILE64_SOURCE \
  -DTAU_BFD \
  -DTAU_MPIFILE \
  -DHAVE_GNU_DEMANGLE \
  -DHAVE_TR1_HASH_MAP \
  -DTAU_SS_ALLOC_SUPPORT \
  -DEBS_CLOCK_RES=1 \
  -DTAU_STRSIGNAL_OK \
  -DTAU_UNWIND \
  -DTAU_USE_LIBUNWIND \
  -I$TAU_ROOT/tau-2.31/x86_64/libunwind-1.3.1-gcc/include \
  -DTAU_TRACK_LD_LOADER \
  -DTAU_OPENMP_NESTED \
  -DTAU_USE_OMPT_5_0 \
  -DTAU_USE_TLS \
  -DTAU_MPICH3 \
  -DTAU_MPI_EXTENSIONS \
  -I$TAU_ROOT/tau-2.31/x86_64/otf2-gcc/include \
  -DTAU_OTF2 \
  -DTAU_ELF_BFD \
  -DTAU_DWARF \
  -I$TAU_ROOT/tau-2.31/x86_64/libdwarf-gcc/include \
  -fopenmp \
  -DTAU_OPENMP \
  -DTAU_UNIFY \
  -O2 \
  -g \
  -fPIC \
  -fPIC \
  -I. \
  -g \
  -c Tau_plugin_$PLUGIN.cc \
  -o Tau_plugin_$PLUGIN.o

g++ \
  -L$TAU_ROOT/tau-2.31/x86_64/lib \
  -lTauMpi-ompt-mpi-pdt-openmp \
  -L$TAU_ROOT/tau-2.31/x86_64/lib \
  -lTauMpi-ompt-mpi-pdt-openmp \
  -Wl,-Bsymbolic-functions \
  -Wl,-z,relro \
  -I/usr/include/x86_64-linux-gnu/mpich \
  -L/usr/lib/x86_64-linux-gnu \
  -lmpichcxx \
  -lmpich \
  -L/usr/lib \
  -Wl,-rpath,/usr/lib \
  -L$TAU_ROOT/tau-2.31/x86_64/lib/shared-ompt-mpi-pdt-openmp \
  -Wl,-rpath,$TAU_ROOT/tau-2.31/x86_64/lib/shared-ompt-mpi-pdt-openmp \
  -lTAU \
  -shared \
  -o libTAU-$PLUGIN_OUT.so Tau_plugin_$PLUGIN.o

/bin/cp libTAU-$PLUGIN_OUT.so $TAU_ROOT/tau-2.31/x86_64/lib/shared-ompt-mpi-pdt-openmp
