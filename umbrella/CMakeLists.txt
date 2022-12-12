#
# Copyright (c) 2021 Carnegie Mellon University,
# Copyright (c) 2021 Triad National Security, LLC, as operator of
#     Los Alamos National Laboratory.
#
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file. See the AUTHORS file for names of contributors.
#

#
# CMakeLists.txt  build carp and its environment
# 12-Mar-2021  chuck@ece.cmu.edu
#

#
# general command line config:
#
#   -DCMAKE_INSTALL_PREFIX=/usr/local     # installation prefix
#   -DCMAKE_BUILD_TYPE=RelWithDebInfo     # or Release, Debug, etc.
#      (XXX: currently only applied to cmake-based builds)
#
#   -DUMBRELLA_BUILDTESTS=OFF             # build unit tests?
#   -DUMBRELLA_RUNTESTS=OFF               # skip running unit tests?
#
# finding dependencies:
#
#   -DCMAKE_PREFIX_PATH='/pkg'            # look for additional installs here
#
# the following also applies for configure scripts:
#   -DCMAKE_INCLUDE_PATH='/pkg/include'   # extra include directories
#   -DCMAKE_LIBRARY_PATH='/pkg/lib'       # extra library path
#
# note these are all cmake lists (so more than one directory can
# be specified using a semicolon to create a path).
#
# specifying alternate compilers (overrides CC/CXX environment variables):
#   -DCC=/bin/cc
#   -DCXX=/bin/cxx
#
#   you may also use the longer names
#     -DCMAKE_C_COMPILER
#     -DCMAKE_CXX_COMPILER
#
# specifying which mpi to use by pointing at the wrappers:
#   -DMPI_C_COMPILER=/usr/bin/mpicc
#   -DMPI_CXX_COMPILER=/usr/bin/mpicxx
#   -DMPIEXEC=/usr/bin/mpiexec
#

cmake_minimum_required (VERSION 3.0)

#
# put the umbrella scripts in our search path and include umbrella-init
# (umbrella-init needs to run before "project")
#
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/umbrella")
include (umbrella-init)

#
# we don't really use MPICXX for now. The difference between CXX and MPICXX
# is that CXX refers to the MPI C API being usable from C++, whereas MPICXX
# refers to the MPI-2 C++ API that was removed again in MPI-3.
# To enable MPICXX, run umbrella with -DMPI_CXX_SKIP_MPICXX=Off.
#
set (MPI_CXX_SKIP_MPICXX ON CACHE BOOL "True if MPICXX should be skipped")

#
# need to mark project as using C/CXX so we can probe for MPI
# using a compiler that may have been passed in via the command line
# or by using environment vars.
#
project (amr-umbrella C CXX)

#
# configure umbrella for MPI and pull in the main routines
#
set (UMBRELLA_MPI 1)
include (umbrella-main)

#
# propagate the MPICXX setting to all cmake subprojects
#
list(APPEND UMBRELLA_CMAKECACHE
     -DMPI_CXX_SKIP_MPICXX:BOOL=${MPI_CXX_SKIP_MPICXX})


#
# set git tags to the versions we want to stick to by default...
#
umbrella_opt_default (DELTAFS_COMMON_TAG "v19.8")
umbrella_opt_default (HDF5_TAG "develop")
umbrella_opt_default (HDF5_ENABLE_PARALLEL ON)

#
# any additional cache variables we want to add that are not already
# part of the umbrella?
#

#
# set other configs before we start pulling in the pieces...
#

set (PDLFS_OPTIONS -DPDLFS_MERCURY_RPC=ON -DPDLFS_SNAPPY=OFF
                   -DPDLFS_GFLAGS=OFF -DPDLFS_GLOG=OFF
                   -DPDLFS_VERBOSE=${DELTAFS_VERBOSE})

include (umbrella/deltafs-common)
include (umbrella/hdf5)
