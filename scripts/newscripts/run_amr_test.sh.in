#!/usr/bin/env bash

set -eu

JOBDIRHOME=/mnt/ltio/amr-runs

# Need to set MPIRUN before invoking common_init
# Need to load mpi_common.sh before accessing amru_prefix
# Call common_init manually to resolve circular dependency
common_noinit=1

arg_test_type="baseline"
arg_host_suffix="dib"
arg_ip_subnet="10.94"
arg_nodes="1"
arg_procs_per_node="16"
arg_cpubind="none"
arg_amr_bin="advection-example"
arg_amr_deck="advection.example.16"
arg_pre="${AMR_PRE-libprof_preload.so}"
arg_nlim="${AMR_NLIM-1}"
arg_lb_policy="${AMR_LB_POLICY-baseline}"

###############
# Core script #
###############

source @CMAKE_INSTALL_PREFIX@/scripts/mpi_common.sh
source ${amru_prefix}/scripts/amr_common.sh

if [ x@AMR_TOOLS_OWNMPI@ != x ]; then
  MPIRUN=$amru_prefix/bin/mpirun
fi

common_init


# load command line args into $arg_* vars (overwrites default values)
loadargs "$@"

procs_per_node="$arg_procs_per_node"
nodes="$arg_nodes"
cores=$((procs_per_node * nodes))
ppn="$procs_per_node"
amr_cpubind="$arg_cpubind"

amr_bin="$arg_amr_bin"
amr_deck="$arg_amr_deck"
lb_policy="$arg_lb_policy"
host_suffix="$arg_host_suffix"

message "Script begin..."
# keep track of start time so we can see how long this takes
timein=`date`

get_jobdir

gen_hosts

amr_prepare_expdir
amr_prepare_deck "$amr_deck" "$lb_policy" "$arg_nlim"
amr_do_run "$arg_pre" "$amr_bin" "$amr_deck"

# overall time
timeout=`date`

message "Script complete!"
message "start: ${timein}"
message "  end: ${timeout}"

exit 0
