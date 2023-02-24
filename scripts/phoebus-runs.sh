#!/bin/bash -eu

. common.sh

process_log() {
  LOG_IN=$1
  LOG_OUT=$LOG_IN.csv
  # LOG_OUT=/dev/stdout

  echo "cycle,time,dt,zc_per_step,wtime_total,wtime_step_other,zc_wamr,wtime_step_amr" > $LOG_OUT
  cat $LOG_IN | grep wsec_AMR | awk -F "\ |=" '{ print $2","$4","$6","$8","$10","$12","$14","$16 }' >> $LOG_OUT
}

tau_linked_build() {
  MPI_HOME=/users/ankushj/amr-workspace/install
  TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004/tau-2.31-profilephase
  HDF5_DIR=/users/ankushj/repos/hdf5/hdf5-2004-psm/CMake-hdf5-1.10.7/HDF5-1.10.7-Linux/HDF_Group/HDF5/1.10.7/share/cmake/hdf5
  PHOEBUS_BUILD_DIR=/users/ankushj/repos/phoebus/build-psm-wtau
  PHOEBUS_BUILD_DIR=/users/ankushj/repos/phoebus/build-psm-wtau-hacks

  mkdir -p $PHOEBUS_BUILD_DIR
  cd $PHOEBUS_BUILD_DIR

  MPI_HOME=$MPI_HOME cmake -DTAU_ROOT=$TAU_ROOT -DHDF5_DIR=$HDF5_DIR ..
}

run_setup_mpi() {
  MPITYPE=mvapich
  MPIRUN=/users/ankushj/amr-workspace/install/bin/mpirun
}

run_setup_amr() {
  AMR_BUILD_BL=/users/ankushj/repos/amr-workspace/phoebus-baseline/build
  AMR_BUILD_HK=/users/ankushj/repos/phoebus/build-psm-wtau
  AMR_BUILD_HK=/users/ankushj/repos/phoebus/build-psm-wtau-hacks-profilephase

  AMR_BUILD=$AMR_BUILD_HK
  # AMR_BUILD=$AMR_BUILD_BL
  AMR_BIN=$AMR_BUILD/src/phoebus
}

#
# run_setup_jobdir:
# uses:
# sets: logfile, exp_logfile, jobdir
# creates: RUN_ROOT, TRACE_ROOT, PROF_ROOT dirs
#

run_setup_jobdir() {
  CLEAN_EXPDIR=${CLEAN_EXPDIR:-0}

  if [[ "$CLEAN_EXPDIR" == "1" ]]; then
    echo "[[ INFO ]] Cleaning exp root: $EXP_ROOT"

    confirm_assume_yes

    rm -rf $EXP_ROOT
  fi

  RUN_ROOT=$EXP_ROOT/run
  TRACE_ROOT=$EXP_ROOT/trace
  PROF_ROOT=$EXP_ROOT/profile

  mkdir -p $RUN_ROOT $TRACE_ROOT $PROF_ROOT

  LOG_FILE="$RUN_ROOT/log.txt"
  LOG_FILE_ALT="$RUN_ROOT/log.alt.txt"

  if [[ -f "$LOG_FILE" ]]; then
    rm $LOG_FILE
  fi

  # set common parameters
  logfile="$LOG_FILE"
  # don't really need this tbh
  exp_logfile="$LOG_FILE_ALT"
  jobdir="$RUN_ROOT"
}

run_setup_env_tau() {
  TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004/tau-2.31-profilephase/x86_64
  TAU_BIN=$TAU_ROOT/bin/tau_exec

  TAU_PLUGINS_PATH=$TAU_ROOT/lib/shared-phase-ompt-mpi-pdt-openmp
  TAU_PLUGINS=libTAU-amr.so

  # With sampling
  TAU=$(echo $TAU_BIN -T ompt,mpi,openmp -ompt -ebs)
  # Without sampling
  # TAU=$(echo $TAU_BIN -T ompt,mpi,openmp)

  # Tau flags
  add_env_var TAU_COMM_MATRIX 1
  add_env_var TAU_TRACK_MESSAGE 1
  add_env_var TAU_PROFILE 1
  add_env_var PROFILEDIR $PROF_ROOT
  add_env_var TAU_CALLPATH 1
  add_env_var TAU_CALLPATH_DEPTH 8
  add_env_var TAU_EBS_UNWIND 1
  add_env_var TAU_EBS_UNWIND_DEPTH 5
  add_env_var TAU_PLUGINS_PATH $TAU_PLUGINS_PATH
  add_env_var TAU_PLUGINS $TAU_PLUGINS
  add_env_var TAU_AMR_LOGDIR $TRACE_ROOT

  CMD_PROFILE="$TAU"
}

run_setup_env_vtune() {
  set +eu
  source /opt/intel/oneapi/vtune/2021.9.0/amplxe-vars.sh
  set -eu

  PROFILER_KNOBS=""
  PROFILER_KNOBS="$PROFILER_KNOBS -k enable-stack-collection=true"
  PROFILER_KNOBS="$PROFILER_KNOBS -k collect-memory-bandwidth=false"

  _CMD="vtune"
  _CMD="$_CMD -collect hpc-performance"
  _CMD="$_CMD -r $PROF_ROOT"
  _CMD="$_CMD -trace-mpi"
  _CMD="$_CMD $PROFILER_KNOBS"
  _CMD="$_CMD --"

  CMD_PROFILE="$_CMD"
}

run_setup_env() {
  ENV_STR=""

  # PSM vars
  add_env_var LD_LIBRARY_PATH /lib64
  add_env_var IPATH_NO_BACKTRACE 1

  # Debug flags
  add_env_var MV2_DEBUG_CORESIZE unlimited

  # Asan flags
  # add_env_var LD_PRELOAD /usr/lib/x86_64-linux-gnu/libasan.so.5
  # add_env_var ASAN_OPTIONS "log_path=/users/ankushj/repos/amr/scripts/asan/asan.log"

  PROFILER_TYPE=${PROFILER_TYPE:-none}

  if [[ "$PROFILER_TYPE" == "tau" ]]; then
    run_setup_env_tau
  elif [[ "$PROFILER_TYPE" == "vtune" ]]; then
    run_setup_env_vtune
  else
    echo "[[ INFO ]] Not using any profiler"
    CMD_PROFILE=""
  fi
}

run_phoebus_cmd() {
  REST_FPATH=${REST_FPATH:-""}

  if [[ "$REST_FPATH" != "" ]]; then
    message "-INFO- Restarting from $REST_FPATH"

    cp $REST_FPATH $RUN_ROOT
    AMR_REST=$(basename $REST_FPATH)
    CMD_AMR=$(echo $AMR_BIN -r $AMR_REST)
  else
    message "-INFO- First Run, Deck: $AMR_DECK"

    CMD_AMR=$(echo $AMR_BIN -i $AMR_DECK)
  fi

  exe="$CMD_PROFILE $CMD_AMR"
  extra_opts=""
  env_vars=( $ENV_STR )

  echo ""

  do_mpirun $procs $ppnode $bind_opt env_vars[@] "$amr_nodes" "$exe" $extra_opts
}

#
# setup_hostfile: sets up local hostfile, detects throttlers, filters them out
# uses: $jobdir, $host_suffix
# sets: $exp_hosts_blacklist
# creates: ./hosts.txt, ./hosts.blacklist.expname
#

setup_hostfile() {
  _exp_name=$(hostname | cut -d. -f 2)
  exp_hosts_blacklist="hosts.blacklist.$_exp_name"

  _jobdir_bak="$jobdir"
  jobdir="."

  gen_hostfile

  message "-INFO- Checking hostfile for throttlers"
  throttling_nodes=$(log_throttling_nodes)

  if [[ "$throttling_nodes" != "" ]]; then
    message "-WARN- Throttlers found: $throttling_nodes. Updating blacklist"
    update_blacklist "$throttling_nodes"

    message "-INFO- Filtering blacklist from hostfile"
    egrep -v -f $exp_hosts_blacklist $jobdir/hosts.txt > tmp
    mv tmp $jobdir/hosts.txt

    message "-INFO- num hosts = $(wc -l ./hosts.txt)"
  else
    message "-INFO- No new throttlers detected."
  fi

  jobdir="$_jobdir_bak"
}

run_phoebus() {
  # -- frequently changed parameters --
  RUN_SUFFIX=profile18
  DRYRUN=0
  CLEAN_EXPDIR=1
  procs=256

  AMR_DECK=/users/ankushj/repos/amr/decks/$RUN_SUFFIX.pin
  AMR_DECK=/users/ankushj/repos/amr/decks/profile14.hacks.pin

  # tau,vtune,none
  PROFILER_TYPE=tau

  # -- relatively static parameters --
  AMR_ROOT=/mnt/ltio/parthenon-topo
  host_suffix="dib"
  ppnode=16
  nodes=$(( procs / ppnode ))
  bind_opt=none

  # -- end of configuration --

  EXP_ROOT=$AMR_ROOT/$RUN_SUFFIX

  run_setup_mpi
  run_setup_amr
  run_setup_jobdir
  run_setup_env

  # Let directories be visible
  sleep 3

  # common_init can only happen now because it needs
  # logfile to be available for logging
  common_init
  setup_hostfile

  cp ./hosts.txt $jobdir
  # generate amr.hosts
  gen_hosts

  run_phoebus_cmd
}

run() {
  # tau_linked_build
  run_phoebus
  # process_log /mnt/ltio/parthenon-topo/profile15/run/log.txt
}

run
