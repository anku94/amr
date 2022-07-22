#!/usr/bin/env bash

set -euxo pipefail

HDF5_ROOT=/users/ankushj/repos/hdf5/hdf5-2004-psm/CMake-hdf5-1.10.7
TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004/tau-2.31
MPI_HOME=/users/ankushj/amr-workspace/install
PHOEBUS_HOME=/users/ankushj/repos/phoebus
PHOEBUS_BUILDDIR=build-psm-wtaucc

HDF5_PATH=$HDF5_ROOT/HDF5-1.10.7-Linux/HDF_Group/HDF5/1.10.7/share/cmake/hdf5
TAU_CC="$TAU_ROOT/x86_64/bin/tau_cc.sh"
TAU_CXX="$TAU_ROOT/x86_64/bin/tau_cxx.sh"
TAU_CXX_COMPILE_OPTS="-optCompInst"

PHOEBUS_BUILDPATH=$PHOEBUS_HOME/$PHOEBUS_BUILDDIR

export MPI_HOME=$MPI_HOME
export TAU_MAKEFILE=$TAU_ROOT/include/Makefile
export TAU_OPTIONS=-optCompInst

build_with_tau_wrappers() {
  CMAKEOPTS=""
  CMAKEOPTS="$CMAKEOPTS -DHDF5_DIR=$HDF5_PATH"
  CMAKEOPTS="$CMAKEOPTS -DCMAKE_C_COMPILER=$TAU_CC"
  CMAKEOPTS="$CMAKEOPTS -DCMAKE_CXX_COMPILER=$TAU_CXX"
  CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_COMPILER=$TAU_CXX"

  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_C_COMPILE_OPTIONS=$TAU_CXX_COMPILE_OPTS"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_C_FLAGS=$TAU_CXX_COMPILE_OPTS"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_C_FLAGS_INIT=$TAU_CXX_COMPILE_OPTS"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_CXX_FLAGS=$TAU_CXX_COMPILE_OPTS"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_CXX_FLAGS_INIT=$TAU_CXX_COMPILE_OPTS"
  # CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_COMPILE_OPTIONS=$TAU_CXX_COMPILE_OPTS"

  CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_LIB_NAMES=mpicxx;mpi"
  CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_HEADER_DIR=$MPI_HOME/include"
  CMAKEOPTS="$CMAKEOPTS -DMPI_mpi_LIBRARY=$MPI_HOME/lib/libmpi.so"
  CMAKEOPTS="$CMAKEOPTS -DMPI_mpicxx_LIBRARY=$MPI_HOME/lib/libmpicxx.so"

  CMAKE_LINKFLAGS=""
}

build_without_tau_wrappers() {
  # CMAKEOPTS=""
  # CMAKEOPTS="$CMAKEOPTS -DHDF5_DIR=$HDF5_PATH"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_C_COMPILER=$TAU_CC"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_CXX_COMPILER=$TAU_CXX"
  # CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_COMPILER=$TAU_CXX"

  # XXX: REMOVED FOPENMP
  CMAKE_DFLAGS="-DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -DTAU_MPI -DTAU_UNIFY -DTAU_MPI_THREADED -DTAU_LINUX_TIMERS -DTAU_MPIGREQUEST -DTAU_MPIDATAREP -DTAU_MPIERRHANDLER -DTAU_MPICONSTCHAR -DTAU_MPIATTRFUNCTION -DTAU_MPITYPEEX -DTAU_MPIADDERROR -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE -DTAU_BFD -DTAU_MPIFILE -DHAVE_GNU_DEMANGLE -DHAVE_TR1_HASH_MAP -DTAU_SS_ALLOC_SUPPORT -DEBS_CLOCK_RES=1 -DTAU_STRSIGNAL_OK -DTAU_UNWIND -DTAU_USE_LIBUNWIND -DTAU_TRACK_LD_LOADER -DTAU_OPENMP_NESTED -DTAU_USE_OMPT_5_0 -DTAU_USE_TLS -DTAU_MPICH3 -DTAU_MPI_EXTENSIONS -DTAU_OTF2 -DTAU_ELF_BFD -DTAU_DWARF -DTAU_OPENMP -DTAU_UNIFY"

  CMAKE_INCL_FLAGS="-I$TAU_ROOT/x86_64/libunwind-1.3.1-mpicc/include -I$TAU_ROOT/x86_64/otf2-mpicc/include -I$TAU_ROOT/x86_64/libdwarf-mpicc/include -I$TAU_ROOT/include"

  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_C_FLAGS='$CXX_COMPILE_OPTS'"
  # CMAKEOPTS="$CMAKEOPTS -DCMAKE_CXX_FLAGS=$CXX_COMPILE_OPTS"
  # CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_COMPILE_OPTIONS=$CXX_COMPILE_OPTS"

  # CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_LIB_NAMES=mpicxx;mpi"
  # CMAKEOPTS="$CMAKEOPTS -DMPI_CXX_HEADER_DIR=$MPI_HOME/include"
  # CMAKEOPTS="$CMAKEOPTS -DMPI_mpi_LIBRARY=$MPI_HOME/lib/libmpi.so"
  # CMAKEOPTS="$CMAKEOPTS -DMPI_mpicxx_LIBRARY=$MPI_HOME/lib/libmpicxx.so"

  CMAKE_LINKFLAGS="-L$TAU_ROOT/x86_64/lib -lTauMpi-ompt-mpi-pdt-openmp -Wl,-rpath,/users/ankushj/repos/amr-workspace/install/lib -L$TAU_ROOT/x86_64/lib -ltau-ompt-mpi-pdt-openmp -L$TAU_ROOT/x86_64/binutils-2.36/lib -L$TAU_ROOT/x86_64/binutils-2.36/lib64 -Wl,-rpath,$TAU_ROOT/x86_64/binutils-2.36/lib -Wl,-rpath,$TAU_ROOT/x86_64/binutils-2.36/lib64 -lbfd -liberty -lz -ldl -Wl,--export-dynamic -lrt -L$TAU_ROOT/x86_64/libunwind-1.3.1-mpicc/lib -lunwind -ldl -Wl,-rpath=$TAU_ROOT/x86_64/lib/shared-ompt-mpi-pdt-openmp -L$TAU_ROOT/x86_64/lib/shared-ompt-mpi-pdt-openmp -lomp -lm -L$TAU_ROOT/x86_64/otf2-mpicc/lib -lotf2 -lotf2 -Wl,-rpath,$TAU_ROOT/x86_64/otf2-mpicc/lib -L$TAU_ROOT/x86_64/libdwarf-mpicc/lib -Wl,-rpath,$TAU_ROOT/x86_64/libdwarf-mpicc/lib -ldwarf -lz -lelf -L$TAU_ROOT/x86_64/lib/static-ompt-mpi-pdt-openmp -lgcc_s"

  # CMAKE_LINKFLAGS="-L$TAU_ROOT/x86_64/lib -lTauMpi-ompt-mpi-pdt-openmp -rpath /users/ankushj/repos/amr-workspace/install/lib -L$TAU_ROOT/x86_64/lib -ltau-ompt-mpi-pdt-openmp -L$TAU_ROOT/x86_64/binutils-2.36/lib -L$TAU_ROOT/x86_64/binutils-2.36/lib64 -rpath $TAU_ROOT/x86_64/binutils-2.36/lib -rpath $TAU_ROOT/x86_64/binutils-2.36/lib64 -lbfd -liberty -lz -ldl --export-dynamic -lrt -L$TAU_ROOT/x86_64/libunwind-1.3.1-mpicc/lib -lunwind -ldl -rpath=$TAU_ROOT/x86_64/lib/shared-ompt-mpi-pdt-openmp -L$TAU_ROOT/x86_64/lib/shared-ompt-mpi-pdt-openmp -lomp -lm -L$TAU_ROOT/x86_64/otf2-mpicc/lib -lotf2 -lotf2 -rpath $TAU_ROOT/x86_64/otf2-mpicc/lib -L$TAU_ROOT/x86_64/libdwarf-mpicc/lib -rpath $TAU_ROOT/x86_64/libdwarf-mpicc/lib -ldwarf -lz -lelf -L$TAU_ROOT/x86_64/lib/static-ompt-mpi-pdt-openmp -lgcc_s"

  # CXX_COMPILE_OPTS="-I$TAU_ROOT/include"
  # CXX_COMPILE_OPTS="$CMAKE_DFLAGS $CMAKE_INCL_FLAGS -fopenmp $CMAKE_LINKFLAGS"
  CXX_COMPILE_OPTS="$CMAKE_DFLAGS $CMAKE_INCL_FLAGS -fopenmp"
}

# build_with_tau_wrappers
build_without_tau_wrappers

rm -rf $PHOEBUS_BUILDPATH/*
cd $PHOEBUS_BUILDPATH
# export $ENVOPTS
# cmake $CMAKEOPTS ..
cmake -DHDF5_DIR=$HDF5_PATH -DCMAKE_EXE_LINKER_FLAGS="$CMAKE_LINKFLAGS" -DCMAKE_C_FLAGS="$CXX_COMPILE_OPTS" -DCMAKE_CXX_FLAGS="$CXX_COMPILE_OPTS" -DMPI_CXX_COMPILE_OPTIONS="$CXX_COMPILE_OPTS" ..
# cmake -DHDF5_DIR=$HDF5_PATH -DCMAKE_C_FLAGS="$CXX_COMPILE_OPTS" -DCMAKE_CXX_FLAGS="$CXX_COMPILE_OPTS" -DMPI_CXX_COMPILE_OPTIONS="$CXX_COMPILE_OPTS" ..
