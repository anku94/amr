#!/usr/bin/env bash

BIN_OMPI=/users/ankushj/repos/parthenon/build-ompi-devspack/example/advection/advection-example
BIN_PSM=/users/ankushj/repos/parthenon/build-psm-dev/example/advection/advection-example

PATH_DECK=/mnt/lustre/parthenon-advection/parthinput.advection
# PATH_DECK=/mnt/lustre/parthenon-advection/parthinput.advection.micro

BIN_TAU=/users/ankushj/repos/tau/tau-2.31/x86_64/bin/tau_exec
BIN_TAU_PSM=/users/ankushj/repos/tau/tau-psm-2.31/tau-2.31/x86_64/bin/tau_exec

OMPI_RUN=mpirun
PSM_RUN=/users/ankushj/repos/tau/mvapich2-install/bin/mpirun

DRYRUN=0

hoststr() {
  cnt=$1
  cnt=$(( cnt - 1 ))
  echo $(seq 0 $cnt | sed 's/^/h/g' | paste -sd, -)
}

load_ompi_spack() {
  . ~/spack/share/spack/setup-env.sh
  spack load mpi
}

run_psm_job() {
  HOSTSTR=$1
  TAUARGS=$2
  LOGFILE=$3

  BIN=$BIN_PSM

  cmd="$PSM_RUN --host $HOSTSTR $TAUARGS $BIN -i $PATH_DECK | tee $LOGFILE"
  echo -e "$cmd\n\n"

  if [[ $DRYRUN -eq 0 ]]; then
    LD_LIBRARY_PATH=/usr/lib64 $cmd
  fi
}

run_ompi_job() {
  IFACE=$1
  HOSTSTR=$2
  TAUARGS=$3
  LOGFILE=$4

  BIN=$BIN_OMPI

  OMPI_FLAGS=" --mca btl tcp,self,vader --mca btl_tcp_if_include $IFACE --mca pml ob1"
  cmd="$OMPI_RUN $OMPI_FLAGS --host $HOSTSTR $TAUARGS $BIN -i $PATH_DECK"

  echo -e "$cmd\n\n"

  if [[ $DRYRUN -eq 0 ]]; then
    $cmd | tee $LOGFILE
  fi
}

run_suite() {
  RUNDIR=$1
  NODECNT=( 2 4 8 )
  NODECNT=( 8 4 2 )

  echo -e "Rundir: $RUNDIR\n\n"

  for cnt in "${NODECNT[@]}"; do
    hstr=$(hoststr $cnt)

    iface=ibs2
    JOBRUNDIR=$RUNDIR/run-ompi-noprof-$cnt-$iface
    [[ $DRYRUN -eq 0 ]] && mkdir -p $JOBRUNDIR
    [[ $DRYRUN -eq 0 ]] && cd $JOBRUNDIR
    run_ompi_job $iface $hstr "" $JOBRUNDIR/log.txt

    JOBRUNDIR=$RUNDIR/run-ompi-tauprof-$cnt-$iface
    [[ $DRYRUN -eq 0 ]] && mkdir -p $JOBRUNDIR
    [[ $DRYRUN -eq 0 ]] && cd $JOBRUNDIR
    run_ompi_job $iface $hstr $BIN_TAU $JOBRUNDIR/log.txt

    iface=eno1
    JOBRUNDIR=$RUNDIR/run-ompi-noprof-$cnt-$iface
    [[ $DRYRUN -eq 0 ]] && mkdir -p $JOBRUNDIR
    [[ $DRYRUN -eq 0 ]] && cd $JOBRUNDIR
    run_ompi_job $iface $hstr "" $JOBRUNDIR/log.txt

    JOBRUNDIR=$RUNDIR/run-ompi-tauprof-$cnt-$iface
    [[ $DRYRUN -eq 0 ]] && mkdir -p $JOBRUNDIR
    [[ $DRYRUN -eq 0 ]] && cd $JOBRUNDIR
    run_ompi_job $iface $hstr $BIN_TAU $JOBRUNDIR/log.txt

    # iface=psm
    # JOBRUNDIR=$RUNDIR/run-psm-noprof-$cnt-$iface
    # [[ $DRYRUN -eq 0 ]] && mkdir -p $JOBRUNDIR
    # [[ $DRYRUN -eq 0 ]] && cd $JOBRUNDIR
    # run_psm_job $hstr "" $JOBRUNDIR/log.txt

    # iface=psm
    # JOBRUNDIR=$RUNDIR/run-psm-tauprof-$cnt-$iface
    # [[ $DRYRUN -eq 0 ]] && mkdir -p $JOBRUNDIR
    # [[ $DRYRUN -eq 0 ]] && cd $JOBRUNDIR
    # run_psm_job $hstr $BIN_TAU_PSM $JOBRUNDIR/log.txt
  done
}

run() {
  BASE_PATH=/mnt/lustre/parthenon-advection
  RUNDIR=$BASE_PATH/runs.$(echo $RANDOM)
  echo -e "\n\nJob base directory: " $RUNDIR "\n"
  mkdir -p $RUNDIR

  load_ompi_spack
  run_suite $RUNDIR
}

run
