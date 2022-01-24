#!/usr/bin/env bash

# Applied to the UBUNTU20-64-AMR image

sudo apt install -y mlocate
sudo updatedb

. ~/spack/share/spack/setup-env.sh
# mpirun (Open MPI) 4.1.2
spack load mpi
# gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0

# ./configure; make -j
cd /users/ankushj/repos/tau/psm
sudo make install

cd /users/ankushj/repos/parthenon/external/Catch2/build
sudo make install

# prebuilt /users/ankushj/repos/parthenon/install/lib/cmake/Kokkos
# prebuilt /users/ankushj/repos/hdf5/hdf5-spack/CMake-hdf5-1.10.7/HDF5-1.10.7-Linux/HDF_Group/HDF5/1.10.7/share/cmake/hdf5
# mvapich: ./configure CC=/users/ankushj/spack/opt/spack/linux-ubuntu20.04-sandybridge/gcc-9.3.0/openmpi-4.1.2-glyuvmslbdbsxbyxweajm6hyi35sesbv/bin/mpicc CXX=/users/ankushj/spack/opt/spack/linux-ubuntu20.04-sandybridge/gcc-9.3.0/openmpi-4.1.2-glyuvmslbdbsxbyxweajm6hyi35sesbv/bin/mpicxx
