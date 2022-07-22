#!/usr/bin/env bash

. common.sh

trap_handler() {
  echo "Trapping SIGINT and propagating..."
  pkill -INT mpirun
  sleep 60
}

openmpi_run_fullmpi() {
  . ~/spack/share/spack/setup-env.sh
  spack env deactivate
  spack load mpi
  JOBDIR=/mnt/lustre/parthenon-phoebus/run-micro
  MPIEXEC=mpirun
  PROG=/users/ankushj/repos/phoebus/build-omp/src/phoebus
  DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_2d_parthreads.pin
  DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d.pin
  DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_2d_parthreads_micro.pin
  export PATH=$PATH:/users/ankushj/repos/tau/tau-ompi-ubuntu2004/tau-2.31/x86_64/bin

  TAU=$(echo /users/ankushj/repos/tau/tau-ompi-ubuntu2004/tau-2.31/x86_64/bin/tau_exec -T ompt,mpi,openmp -ompt -ebs)

  cmd=$(echo $MPIEXEC $(binding_flags) --host $(hoststr 8) $(mca_flags) $PROG -i $DECK)
  cmd=$(echo $MPIEXEC $(binding_flags) --host $(hoststr 8) $(mca_flags) $TAU $PROG -i $DECK)
  cmd=$(echo $MPIEXEC $(binding_flags) $(tau_flags) --host $(hoststr 8) $(mca_flags) $TAU $PROG -i $DECK)
  cmd=$(echo $MPIEXEC $(tau_flags) --host $(hoststr 8) $(mca_flags) $TAU $PROG -i $DECK)
  # cmd=$(echo $MPIEXEC --host $(hoststr 8) $(mca_flags) $PROG -i $DECK)
  echo $cmd

  cd $JOBDIR
  $cmd 2>&1 | tee log.txt
}

openmpi_run() {
  JOBDIR=/mnt/lustre/parthenon-phoebus/run-3d-128-1t1288p-reflvl2
  mkdir -p $JOBDIR
  rm -rf $JOBDIR/*

  MPIEXEC=mpirun
  PROG=/users/ankushj/repos/phoebus/build-mpionly/src/phoebus
  PROG=/users/ankushj/repos/phoebus/build-mpionly-2/src/phoebus
  # PROG=/users/ankushj/repos/phoebus/build-omp-4t/src/phoebus
  # PROG=/users/ankushj/repos/phoebus/build-omp-8t/src/phoebus
  # PROG=/users/ankushj/repos/phoebus/build-omp-16t/src/phoebus
  # PROG=/users/ankushj/repos/phoebus/build-omp/parthenon/example/calculate_pi/pi-example
  DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_2d_parthreads.pin
  DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d.pin
  DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_64.pin
  # DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_2d_parthreads_micro.pin
  # DECK=/users/ankushj/repos/parthenon/example/calculate_pi/parthinput.example.big
  export PATH=$PATH:/users/ankushj/repos/amr-workspace/tau/tau-2.31/x86_64/bin

  TAU=$(echo /users/ankushj/repos/amr-workspace/tau/tau-2.31/x86_64/bin/tau_exec -T ompt,mpi,openmp -ompt -ebs)

  cmd=$(echo $MPIEXEC $(binding_flags) --host $(hoststr 8) $(mca_flags) $PROG -i $DECK)
  cmd=$(echo $MPIEXEC $(binding_flags) --host $(hoststr 8) $(mca_flags) $TAU $PROG -i $DECK)
  cmd=$(echo $MPIEXEC $(binding_flags) $(tau_flags) --host $(hoststr 8 1) -np 8 $(mca_flags) $TAU $PROG -i $DECK)
  # cmd=$(echo $MPIEXEC $(tau_flags) --host $(hoststr 8) -np 128 --map-by ppr:1:core $(mca_flags) $TAU $PROG -i $DECK)
  cmd=$(echo $MPIEXEC $(tau_flags) --host $(hoststr 8 16) -np 128 --map-by ppr:1:core $(mca_flags) $TAU $PROG -i $DECK)

  echo $cmd

  cd $JOBDIR
  $cmd 2>&1 | tee log.txt
}

tau_plugin_run() {
  AMR_BIN=/users/ankushj/repos/phoebus/build-mpich-1t/src/phoebus
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_micro.pin
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_64.pin
  RUN_ROOT=/mnt/lustre/parthenon-phoebus/tauplugtest-psm
  mkdir -p $RUN_ROOT
  cp hosts.txt $RUN_ROOT
  cd $RUN_ROOT

  TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-mpich-2004
  TAU_REL=tau-2.31/x86_64/bin/tau_exec
  TAU_BIN=$TAU_ROOT/$TAU_REL
  TAU_PLUGINS_PATH=/users/ankushj/repos/amr-workspace/tau-mpich-2004/tau-2.31/x86_64/lib/shared-ompt-mpi-pdt-openmp
  TAU_PLUGINS=libTAU-amr.so
  echo $TAU_BIN

  TAU=$(echo $TAU_BIN -T ompt,mpi,openmp -ompt -ebs)
  TAU_FLAGS=$(tau_flags_mpich)
  TAU_FLAGS=$(echo $TAU_FLAGS -env TAU_PLUGINS_PATH $TAU_PLUGINS_PATH -env TAU_PLUGINS $TAU_PLUGINS)
  # TAU_FLAGS=

  mpirun -f hosts.txt -np 512 -map-by ppr:16:node $TAU_FLAGS $TAU $AMR_BIN -i $AMR_DECK
}

reg_mpich_run() {
  AMR_BIN=/users/ankushj/repos/phoebus/build-mpich-1t/src/phoebus
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_micro.pin
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_64.pin

  RUN_ROOT=/mnt/lustre/parthenon-phoebus/notautest
  mkdir -p $RUN_ROOT
  cp hosts.txt $RUN_ROOT
  cd $RUN_ROOT

  mpirun -f hosts.txt -np 512 -map-by ppr:16:node $TAU_FLAGS $TAU $AMR_BIN -i $AMR_DECK
}

reg_mvapich_run() {
  AMR_BIN=/users/ankushj/repos/phoebus/build-psm/src/phoebus
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_micro.pin
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_64.pin

  MPIRUN=/users/ankushj/amr-workspace/install/bin/mpirun

  RUN_ROOT=/mnt/lustre/parthenon-phoebus/notautest
  mkdir -p $RUN_ROOT
  cp hosts.txt $RUN_ROOT
  cd $RUN_ROOT

  envstr="-env LD_LIBRARY_PATH /lib64"

  cmd=$(echo $MPIRUN -f hosts.txt $envstr -np 128 -map-by ppr:16:node $AMR_BIN -i $AMR_DECK)
  echo $cmd
  $cmd
}

tau_mvapich_run() {
  AMR_BIN=/users/ankushj/repos/phoebus/build-psm/src/phoebus
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_micro.pin
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_64.pin

  MPIRUN=/users/ankushj/amr-workspace/install/bin/mpirun

  EXP_ROOT=/mnt/ltio/parthenon-topo/profile6.wtau
  RUN_ROOT=$EXP_ROOT/run
  TRACE_ROOT=$EXP_ROOT/trace

  mkdir -p $RUN_ROOT $TRACE_ROOT
  cp hosts.txt $RUN_ROOT
  cd $RUN_ROOT

  TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004
  TAU_REL=tau-2.31/x86_64/bin/tau_exec
  TAU_BIN=$TAU_ROOT/$TAU_REL
  TAU_PLUGINS_PATH=$TAU_ROOT/tau-2.31/x86_64/lib/shared-ompt-mpi-pdt-openmp
  TAU_PLUGINS=libTAU-amr.so
  echo $TAU_BIN

  TAU=$(echo $TAU_BIN -T ompt,mpi,openmp -ompt -ebs)
  TAU=$(echo $TAU_BIN -T ompt,mpi,openmp -ompt -ebs)
  TAU_FLAGS=$(tau_flags_mpich)
  TAU_FLAGS=$(echo $TAU_FLAGS -env TAU_PLUGINS_PATH $TAU_PLUGINS_PATH -env TAU_PLUGINS $TAU_PLUGINS)

  envstr="-env LD_LIBRARY_PATH /lib64 $TAU_FLAGS"
  envstr="$envstr"


  # mpirun -f hosts.txt $envstr -np 512 -map-by ppr:16:node $TAU_FLAGS $TAU $AMR_BIN -i $AMR_DECK
  cmd=$(echo $MPIRUN -f hosts.txt $envstr -np 512 -map-by ppr:16:node $TAU $AMR_BIN -i $AMR_DECK)
  # cmd=$(echo $MPIRUN $envstr -np 16 $TAU $AMR_BIN -i $AMR_DECK)
  rm log.txt; touch log.txt
  echo $cmd | tee -a log.txt
  $cmd 2>&1 | tee -a log.txt
  process_log log.txt
}

process_log() {
  LOG_IN=$1
  LOG_OUT=$LOG_IN.csv
  # LOG_OUT=/dev/stdout

  echo "cycle,time,dt,zc_per_step,wtime_total,wtime_step_other,zc_wamr,wtime_step_amr" > $LOG_OUT
  cat $LOG_IN | grep wsec_AMR | awk -F "\ |=" '{ print $2","$4","$6","$8","$10","$12","$14","$16 }' >> $LOG_OUT
}

process_profile() {
  PROFDIR=/mnt/ltio/parthenon-topo/profile6/run

  cd $PROFDIR

  PPROF_BIN=/users/ankushj/repos/amr-workspace/tau/tau-2.31/x86_64/bin/pprof

  PROFLOG=profile.log.csv

  echo rank,event,timepct | tee $PROFLOG
  for i in `seq 0 511`; do
    # time_pct=$($PPROF_BIN -a -m $i | grep 'UpdateMeshBlockTree => MPI_Allgather()\s*$' | grep 'Driver_Main' | awk '{ print $1 }')
    time_pct_a=$($PPROF_BIN -a -m $i | grep 'UpdateMeshBlockTree => MPI_Allgather()\s*$' | grep 'Driver_Main' | awk '{ print $1 }')
    time_pct_b=$($PPROF_BIN -a -m $i | grep 'Task_FillDerived => FillDerived => ConToPrim::Solve\s*$' | grep 'Driver_Main' | awk '{ print $1 }')
    time_pct_c=$($PPROF_BIN -a -m $i | grep '.TAU application => Driver_Main => Task_ReceiveBoundaryBuffers_MeshData => Task_ReceiveBoundaryBuffers_MeshBlockData\s*$' | grep 'Driver_Main' | awk '{ print $1 }')
    echo $i,MPIAllGather,$time_pct_a | tee -a $PROFLOG
    echo $i,ConToPrimeSolve,$time_pct_b | tee -a $PROFLOG
    echo $i,ReceiveBoundaryBufs,$time_pct_c | tee -a $PROFLOG
  done
}

tau_linked_build() {
  MPI_HOME=/users/ankushj/amr-workspace/install
  TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004/tau-2.31
  HDF5_DIR=/users/ankushj/repos/hdf5/hdf5-2004-psm/CMake-hdf5-1.10.7/HDF5-1.10.7-Linux/HDF_Group/HDF5/1.10.7/share/cmake/hdf5
  PHOEBUS_BUILD_DIR=/users/ankushj/repos/phoebus/build-psm-wtau

  mkdir -p $PHOEBUS_BUILD_DIR
  cd $PHOEBUS_BUILD_DIR

  MPI_HOME=$MPI_HOME cmake -DTAU_ROOT=$TAU_ROOT -DHDF5_DIR=$HDF5_DIR ..
} 

tau_linked_run() {
  AMR_BIN=/users/ankushj/repos/phoebus/build-psm-wtau/src/phoebus
  # AMR_BIN=/users/ankushj/repos/phoebus/build-psm/src/phoebus
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_micro.pin
  AMR_DECK=/users/ankushj/repos/phoebus/inputs/blast_wave_3d_64.pin

  MPIRUN=/users/ankushj/amr-workspace/install/bin/mpirun

  EXP_ROOT=/mnt/ltio/parthenon-topo/profile8
  RUN_ROOT=$EXP_ROOT/run
  TRACE_ROOT=$EXP_ROOT/trace
  PROF_ROOT=$EXP_ROOT/profile

  mkdir -p $RUN_ROOT $TRACE_ROOT $PROF_ROOT
  cp hosts.txt $RUN_ROOT
  cd $RUN_ROOT

  TAU_ROOT=/users/ankushj/repos/amr-workspace/tau-psm-2004
  TAU_REL=tau-2.31/x86_64/bin/tau_exec
  TAU_BIN=$TAU_ROOT/$TAU_REL
  TAU_PLUGINS_PATH=$TAU_ROOT/tau-2.31/x86_64/lib/shared-ompt-mpi-pdt-openmp
  TAU_PLUGINS=libTAU-amr.so
  # TAU_PLUGINS=libTAU-load-balance.so
  echo $TAU_BIN

  TAU=$(echo $TAU_BIN -T ompt,mpi,openmp -ompt -ebs)
  # Disable sampling
  TAU=$(echo $TAU_BIN -T ompt,mpi,openmp)
  TAU_FLAGS=$(tau_flags_mpich)
  TAU_FLAGS=$(echo $TAU_FLAGS -env TAU_PLUGINS_PATH $TAU_PLUGINS_PATH -env TAU_PLUGINS $TAU_PLUGINS -env TAU_AMR_LOGDIR $TRACE_ROOT)

  envstr="-env LD_LIBRARY_PATH /lib64 $TAU_FLAGS"
  # envstr="$envstr -env LD_PRELOAD /usr/lib/x86_64-linux-gnu/libasan.so.5 -env ASAN_OPTIONS log_path=/users/ankushj/repos/amr/scripts/asan/asan.log"
  # envstr="$envstr -env LD_PRELOAD /usr/lib/x86_64-linux-gnu/libasan.so.5"
  envstr="$envstr"


  # head -8 hosts.txt > hosts8.txt
  cmd=$(echo $MPIRUN -f hosts.txt $envstr -np 512 -map-by ppr:16:node $TAU_FLAGS $TAU $AMR_BIN -i $AMR_DECK)
  # cmd=$(echo $MPIRUN -f hosts.txt $envstr -np 512 -map-by ppr:16:node $AMR_BIN -i $AMR_DECK)
  # cmd=$(echo $MPIRUN $envstr -np 4 -map-by ppr:16:node $TAU_FLAGS $TAU $AMR_BIN -i $AMR_DECK)
  # cmd=$(echo $MPIRUN $envstr -np 4 -map-by ppr:16:node $TAU_FLAGS $AMR_BIN -i $AMR_DECK)
  # rm log.txt; touch log.txt

  echo $cmd | tee -a log.txt
  # $cmd 2>&1 | tee -a log.txt
  process_log log.txt
}

# tau_plugin_run
# reg_mpich_run
# reg_mvapich_run
# tau_mvapich_run
# process_log /mnt/ltio/parthenon-topo/profile5/run/log.txt
# process_profile
# tau_linked_build
# tau_linked_run
process_log /mnt/ltio/parthenon-topo/profile8/run/log.txt
