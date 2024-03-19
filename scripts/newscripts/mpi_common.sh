#!/bin/bash -eu

# A stripped down version of pdlfs-scripts/common.sh

# global variables we set/use:
#  $amru_prefix - deltafs umbrella prefix directory (abspath)
#  $common_noinit - set to non-null disable call to common_init (for dbg)
#  $host_suffix - append "-suffix" to hostnames if set - emulab only (string)
#  $ip_subnet - the ip subnet we want to use (x.y.z.t)
#  $jobdir - per-job shared output directory (abspath)
#  $jobenv - job environment (moab, slurm, etc.) (string)
#  $jobruncli - interface to job run command (string)
#  $jobruncmd - job run command (string)
#  $openmpi_newbind - 1 if openmpi mpirun supports "--bind-to"
#  $logfile - log shared by all exp runs (abspath)
#  $exp_logfile - log used by one specific exp run (abspath)
#  $nodes - number of nodes for vpic (int)
#  $bbos_buddies - number of nodes for bbos (int)
#  $cores - total cores across all nodes (int)
#  $all_nodes - list of all nodes (string - sep: comma)
#  $num_all_nodes - number of nodes in all_nodes (int)
#  $vpic_nodes - list of nodes for vpic (string - sep: comma)
#  $num_vpic_nodes - number of nodes in vpic_nodes (int)
#  $bbos_nodes - list of nodes for bbos (string - sep: comma)
#  $num_bbos_nodes - number of nodes in bbos_nodes (int)
#  $bb_log_size - BBOS max per-core log size in bytes
#

# environment variables we set/use:
#  $JOBDIRHOME - where to put job dirs (default: $HOME/jobs)
#                example: /lustre/ttscratch1/users/$USER
#  $EXTRA_MPIOPTS - additional options that need to be passed to mpirun
#  $DW_SESSION_OVERRIDE - (cray) override the datawarp sessionid (for dbg)
#

#
# environment variables we use as input:
#  $HOME - your home directory
#  $DW_JOB_STRIPED - data warp mount (cray)
#  $JOBENV - operating env (one of moab, slurm, cobalt, openmpi, or mpich)
#    if $JOBENV is set, we use:
#       $JOBRUNCLI - command line interface to job run command
#       $JOBRUNCMD - command to run jobs
#  $JOBHOSTS - nodes to run jobs (default: localhost)
#              (only used if not running on a Cray or an Emulab platform)
#  $MPIRUN - mpirun command to use (if not running under a scheduler)
#
#  Env running moab:
#    $MOAB_JOBNAME - jobname
#    $PBS_JOBID - job id
#    $PBS_NODEFILE - file with list of all nodes
#
#  Env running slurm:
#    $SLURM_JOB_NAME - jobname
#    $SLURM_JOBID - job id
#    $SLURM_JOB_NODELIST - list of all nodes (in compat form)
#       XXX: we use the slurm_nodefile script to expand the node list
#
#  Env running cobalt:
#    $COBALT_JOBID - job id
#    $COBALT_NODEFILE - list of all nodes in a file (not always provided)
#    $COBALT_PARTNAME - list of all nodes (in compact form)
#    $COBALT_XJOBNAME - job name (we provide it, optional)
#

#
# files we create:
#  $jobdir/hosts.txt - list of hosts
#  $jobdir/bbos.hosts - host file only of bbos hosts
#  $jobdir/vpic.hosts - host file only of vpic hosts
#

# TODO:
# - Convert node lists to ranges on CRAY

#
# prefix directory comes from cmake's ${CMAKE_INSTALL_PREFIX} variable
#
amru_prefix=@CMAKE_INSTALL_PREFIX@

### ensure definition of a set of global vars ###

#
# job-wise log - shared among all exp runs
# default: null (XXX: but we override this in get_jobdir)
#
logfile=${logfile-}

#
# exp-wise log - one per exp run
# default: null
#
exp_logfile=${exp_logfile-}

#
# common_init: init the common.sh layer (mainly detecting env)
# uses/sets: $JOBENV, $JOBRUNCMD, $JOBRUNCLI, $MPIRUN,
# $jobenv, $jobruncmd, $jobruncli
#

common_init() {

  #
  # the priority ordering of things here is:
  #  1. allow users to hardwire the jobenv using JOBENV
  #  2. look for job ID#s from a scheduler (moab, slurm, cobalt) to do setup
  #  3. look to see if we are already running under openmpi or mpich
  # otherwise we look for a bare MPI environment like this:
  #  4. allow users to use $MPIRUN to specify a specific MPI to use
  #  5. look for mpirun.mpich
  #  6. look for mpirun.openmpi
  #  7. look for plain mpirun and see if it is mpich or openmpi
  #  8. give up
  #
  if [ x${JOBENV-} != x ]; then # allow user to hardwire a jobenv
    if [ $JOBENV != moab -a $JOBENV != slurm -a $JOBENV != cobalt -a \
      $JOBENV != mpich -a $JOBENV != openmpi ]; then
      die "bad JOBENV ($JOBENV) provided by user - ABORTING"
    fi
    if [ x${JOBRUNCLI-} = x -o x${JOBRUNCMD-} = x ]; then
      die "JOBRUNCLI and JOBRUNCMD must be specified with JOBENV - ABORTING"
    fi
    jobenv=$JOBENV jobruncli=$JOBRUNCLI jobruncmd=$JOBRUNCMD
  elif [ x${PBS_JOBID-} != x ]; then
    jobenv=moab jobruncli=aprun jobruncmd=aprun
  elif [ x${SLURM_JOBID-} != x ]; then
    jobenv=slurm jobruncli=srun jobruncmd=srun
  elif [ x${COBALT_JOBID-} != x ]; then
    jobenv=cobalt
    if [ $(which aprun 2> /dev/null) ]; then
      jobruncli=aprun jobruncmd=aprun
    elif [ $(which runjob 2> /dev/null) ]; then
      jobruncli=runjob jobruncmd=runjob
    elif [ $(which mpirun 2> /dev/null) ]; then
      jobruncmd=mpirun
      if [ "$(mpirun --version 2>&1 | fgrep 'open-mpi.org')" ]; then
        jobruncli=openmpi
      else
        jobruncli=mpich
      fi
    else
      die "cobalt: cannot determine job cli/cmd - ABORTING!"
    fi
  elif [ x${OMPI_COMM_WORLD_SIZE-} != x ]; then # since ompi 1.3
    jobenv=openmpi jobruncli=openmpi
    if [ $(which mpirun.openmpi 2> /dev/null) ]; then
      jobruncmd=mpirun.openmpi
    else
      jobruncmd=mpirun
    fi
  elif [ x${HYDI_CONTROL_FD-} != x ]; then # XXX maybe
    jobenv=mpich jobruncli=mpich
    if [ $(which mpirun.mpich 2> /dev/null) ]; then
      jobruncmd=mpirun.mpich
    else
      jobruncmd=mpirun
    fi
  elif [ x${MPIRUN-} != x ]; then
    if [ "$(${MPIRUN} --version 2>&1 | fgrep 'open-mpi.org')" ]; then
      jobenv=openmpi jobruncli=openmpi jobruncmd=${MPIRUN}
    elif [ "$(${MPIRUN} --version 2>&1 | fgrep 'HYDRA')" ]; then
      jobenv=mpich jobruncli=mpich jobruncmd=${MPIRUN}
    else
      die "MPIRUN: cannot determine MPI type for cmd $MPIRUN - ABORTING"
    fi
  elif [ $(which mpirun.mpich 2> /dev/null) ]; then
    jobenv=mpich jobruncli=mpich jobruncmd=mpirun.mpich
  elif [ $(which mpirun.openmpi 2> /dev/null) ]; then
    jobenv=openmpi jobruncli=openmpi jobruncmd=mpirun.openmpi
  elif [ $(which mpirun 2> /dev/null) ]; then
    if [ "$(mpirun --version 2>&1 | fgrep 'open-mpi.org')" ]; then
      jobenv=openmpi jobruncli=openmpi jobruncmd=mpirun
    else
      jobenv=mpich jobruncli=mpich jobruncmd=mpirun
    fi
  else
    die "common.sh UNABLE TO DETERMINE ENV - ABORTING"
  fi

  message "-INFO- common_init: JOBENV=$jobenv, CLI=$jobruncli, CMD=$jobruncmd"

  #
  # check for newer cpu binding flag for openmpi
  #
  if [ $jobruncli = openmpi ]; then
    if [ "$($jobruncmd --help 2>&1 | fgrep -e '--bind-to ')" ]; then
      message "-INFO- common_init: openmpi supports new --bind-to flag"
      openmpi_newbind=1
    else
      message "-INFO- common_init: openmpi does not support new --bind-to flag"
      openmpi_newbind=0
    fi
  fi

  #
  # do some sanity checks
  #
  if [ $jobenv = moab ]; then
    if [ x${PBS_JOBID-} = x -o x${MOAB_JOBNAME-} = x \
      -o ! -f "$PBS_NODEFILE" ]; then
      die "bad moab setup - check jobname and nodefile"
    fi
  elif [ $jobenv = slurm ]; then
    if [ x${SLURM_JOBID-} = x -o x${SLURM_JOB_NAME-} = x -o \
      x${SLURM_JOB_NODELIST-} = x ]; then
      die "bad slurm setup - check jobname and nodelist"
    fi
  elif [ $jobenv = cobalt ]; then
    # cobalt does not give us job name...
    # make one up and ensure it is always in the env at COBALT_XJOBNAME
    if [ x${COBALT_XJOBNAME-} = x ]; then
      export COBALT_XJOBNAME=$(basename $0)
      message "-INFO- common_init: cobalt: set xjobname=${COBALT_XJOBNAME}"
    fi
    if [ x${COBALT_JOBID-} = x ]; then
      die "bad cobalt setup - check jobid"
    fi
    if [ x${COBALT_PARTNAME-} = x -a x${COBALT_NODEFILE-} = x ]; then
      die "bad cobalt setup - check partname/nodefile"
    fi
  fi
}

#
# message: echo message to stdout, cc it to a default job-wise log file, and
# then cc it again to a specific exp-wise log file.
# note that if either $logfile or $exp_logfile is empty, tee will
# just print the message without writing to files
# uses: $logfile, $exp_logfile
#
message() { echo "$@" | tee -a $exp_logfile | tee -a $logfile; }

#
# die: emit a mesage and exit 1
#
die() {
  message "!!! ERROR !!! $@"
  exit 1
}

#
# loadargs(): parse a bunch of key value pairs into "arg_*" variables
#
loadargs() {
	for loadargs_arg; do
		loadargs_key=$(echo $loadargs_arg | sed -n 's/=.*//p')
		# make sure key is alphanumeric or _
		loadargs_check=$(echo $loadargs_key | sed -e 's/[A-Za-z0-9_]//g')
		loadargs_val=$(echo $loadargs_arg | sed -n 's/^[A-Za-z0-9_]*=//p')
		if [ "$loadargs_check" != "" -o "$loadargs_key" = "" ]; then
			echo "loadargs: bad keyval arg: $loadargs_arg"
			exit 1
		fi
		eval "arg_${loadargs_key}=\"${loadargs_val}\""
	done
}

#
# jobdir  ## Lustre
#
# get_jobdir: setup $jobdir var and makes sure $jobdir is present
# uses: $jobenv, $MOAB_JOBNAME, $PBS_JOBID, $SLURM_JOB_NAME, $SLURM_JOBID,
#       $COBALT_XJOBNAME, $COBALT_JOBID, $JOBDIRHOME
# sets: $jobdir  (XXX: also $logfile)
#
get_jobdir() {
    if [ x${JOBDIRHOME-} != x ]; then
        jobdirhome=${JOBDIRHOME}
    else
        jobdirhome=${HOME}/jobs
    fi

    if [ $jobenv = moab ]; then
        jobdir=${jobdirhome}/${MOAB_JOBNAME}.${PBS_JOBID}
    elif [ $jobenv = slurm ]; then
        jobdir=${jobdirhome}/${SLURM_JOB_NAME}.${SLURM_JOBID}
    elif [ $jobenv = cobalt ]; then
        jobdir=${jobdirhome}/${COBALT_XJOBNAME}.${COBALT_JOBID}
    elif [ x${MPIJOBNAME-} != x ]; then
        jobdir=${jobdirhome}/${MPIJOBNAME}.${MPIJOBID-$$}
    else
        ## TODO: UNCOMMENT THIS
        # jobdir=${jobdirhome}/`basename $0`.$$  # use top-level script name $0
        jobdir=${jobdirhome}/`basename $0`.1234
    fi

    message "-INFO- creating jobdir..."
    mkdir -p ${jobdir} || die "cannot make jobdir ${jobdir}"
    message "-INFO- jobdir = ${jobdir}"

    # XXX: override default and auto set logfile
    logfile=${jobdir}/$(basename $jobdir).log
}

#
# all_nodes
#
# gen_hostfile: generate a list of hosts we have in $jobdir/hosts.txt
# one host per line.
# uses: $jobenv, $PBS_NODEFILE
#       $SLURMJOB_NODELIST/slurm_nodefile,
#       $COBALT_PARTNAME/cobalt_nodefile,
#       $jobdir
# sets: $all_nodes, $num_all_nodes
# creates: $jobdir/hosts.txt
#
gen_hostfile() {
  message "-INFO- generating hostfile ${jobdir}/hosts.txt..."

  if [ $jobenv = moab ]; then

    # Generate hostfile on CRAY and store on disk
    cat $PBS_NODEFILE | uniq | sort > $jobdir/hosts.txt ||
      die "failed to create hosts.txt file"

  elif [ $jobenv = slurm ]; then

    # generate hostfile under slurm with helper script
    $amru_prefix/scripts/slurm_nodefile $jobdir/hosts.txt ||
      die "failed to create hosts.txt file"

  elif [ $jobenv = cobalt ]; then

    # generates hostfile from PARTNAME or NODEFILE
    $amru_prefix/scripts/cobalt_nodefile $jobdir/hosts.txt ||
      die "failed to create hosts.txt file"

  elif [ -f "/share/testbed/bin/emulab-listall" ]; then

    # Generate hostfile on Emulab and store on disk
    if [ x${host_suffix-} != x ]; then
      exp_hosts="$(/share/testbed/bin/emulab-listall --append $host_suffix | tr ',' '\n')"
    else
      exp_hosts="$(/share/testbed/bin/emulab-listall | tr ',' '\n')"
    fi

    echo "$exp_hosts" > $jobdir/hosts.txt ||
      die "failed to create hosts.txt file"
  else

    message "!!! WARNING !!! CRAY or Emulab not available, reading @ENV[JOBHOSTS]"

    echo "${JOBHOSTS:-localhost}" | tr ',' '\n' > $jobdir/hosts.txt ||
      die "failed to create hosts.txt file"
  fi

  if [ ${exp_hosts_blacklist:-"none"} != "none" ] && [ -f "$exp_hosts_blacklist" ]; then
    message "-INFO- removing hosts in $exp_hosts_blacklist from hosts.txt"
    egrep -v -f $exp_hosts_blacklist $jobdir/hosts.txt > tmp
    mv tmp $jobdir/hosts.txt
  fi

  # Populate a variable with hosts
  all_nodes=$(cat ${jobdir}/hosts.txt)
  num_all_nodes=$(cat ${jobdir}/hosts.txt | tr ',' '\n' | sort | wc -l)
  message "-INFO- num hosts = ${num_all_nodes}"
}

#
# generate host list files: $jobdir/amr.hosts, $jobdir/bbos.hosts
# uses: $jobdir, $nodes, $bbos_buddies
# sets: $amr_nodes, $bbos_nodes
# creates: $jobdir/amr.hosts, $jobdir/bbos.hosts
#
gen_hosts() {
  message "-INFO- generating amr/bbos host lists..."

  # XXX: sanity check: # of nodes in list from PBS_NODEFILE or
  # XXX:               emulab-listall >= $nodes+$bbos_buddies

  # first generate the generic hosts.txt
  gen_hostfile

  # XXX AJ: giving up hostifle generation because
  # we added hostfile generation AND filtering
  # outside this function

  # divide hosts.txt into parts based on $nodes and $bbos_nodes
  cat $jobdir/hosts.txt | head -n $nodes |
    tr '\n' ',' | sed '$s/,$//' > $jobdir/amr.hosts ||
    die "failed to create amr.hosts file"

  amr_nodes=$(cat ${jobdir}/amr.hosts)
  num_amr_nodes=$(cat ${jobdir}/amr.hosts | tr ',' '\n' | sort | wc -l)
  message "-INFO- num amr nodes = ${num_amr_nodes}"

  if [ "${bbos_buddies:-0}" = "0" ]; then
    message "-INFO- supressing zero length bbos.hosts"
    bbos_nodes=""
    num_bbos_nodes=0
  else
    cat $jobdir/hosts.txt | tail -n $bbos_buddies |
      tr '\n' ',' | sed '$s/,$//' > $jobdir/bbos.hosts ||
      die "failed to create bbos.hosts file"
    bbos_nodes=$(cat ${jobdir}/bbos.hosts)
    num_bbos_nodes=$(cat ${jobdir}/bbos.hosts | tr ',' '\n' | sort | wc -l)
    message "-INFO- num bbos nodes = ${num_bbos_nodes}"
  fi
}

#
# clear_caches: clear node caches on amr nodes
# uses: $jobenv, $amr_nodes, $cores, $nodes
#
clear_caches() {
  message "-INFO- clearing node caches..."

  if [ $jobenv = moab -o $jobenv = slurm -o $jobenv = cobalt ]; then
    message "!!! NOTICE !!! skipping cache clear ... no hpc sudo access"
  elif [ -f "/share/testbed/bin/emulab-mpirunall" -a "${DROP_CACHE-}x" != "x" ]; then
    # this does more than just $amr_nodes (does them all)
    # but that isn't really a problem...
    /share/testbed/bin/emulab-mpirunall sudo sh -c \
      'echo 3 > /proc/sys/vm/drop_caches'
  else
    message "!!! NOTICE !!! skipping cache clear ... not on Emulab"
  fi

  message "-INFO- done"
}

#
# do_mpirun: Run slurm/cobalt/mpich/openmpi command
#
# Arguments:
# @1 number of processes
# @2 number of processes per node
# @3 cpu binding mode (thread, core, external, none, ...)
# @4 array of env vars: ("name1", "val1", "name2", ... )
# @5 host list (comma-separated)
# @6 executable (and any options that don't fit elsewhere)
# @7 extra_opts: extra options to mpiexec (optional)
# @8 log1: primary log file for mpi stdout (optional)
# @9 log2: secondary log file for mpi stdout (optional)
do_mpirun() {
  procs=$1
  ppnode=$2
  bind_opt="$3"
  if [ ! -z "$4" ]; then
    declare -a envs=(${!4-})
  else
    envs=()
  fi
  hosts="$5"
  exe="$6"
  ### extra options to mpiexec ###
  extra_opts=${7-}
  ### log files ###
  log1=${8-$logfile}
  log2=${9-$exp_logfile}

  envstr=""
  npstr=""
  hstr=""

  if [ $jobruncli = aprun ]; then

    # CRAY with aprun (could be moab or cobalt)
    if [ ${#envs[@]} -gt 0 ]; then
      envstr=$(printf -- "-e %s=%s " ${envs[@]})
    fi

    if [ $ppnode -gt 0 ]; then
      npstr="-N $ppnode"
    fi

    if [ ! -z "$hosts" ]; then
      hstr="-L $hosts"
    fi

    if [ x$bind_opt = xthread ]; then
      bind_opt="-cc cpu"
    elif [ x$bind_opt = xcore ]; then
      message "do_mpirun: aprun: converted core CPU binding to thread"
      bind_opt="-cc cpu"
    elif [ x$bind_opt = xexternal ]; then
      bind_opt=""
    elif [ x$bind_opt = xnone -o x$bind_opt = x ]; then
      bind_opt="-cc none"
    else
      die "do_mpirun: bad bind ops $bind_opt"
    fi

    message "[MPIEXEC]" "$jobruncmd -q -n $procs" $npstr $hstr $envstr \
      $bind_opt $extra_opts ${DEFAULT_MPIOPTS-} $exe

    if [ x${DRYRUN-0} != x0 ]; then
      return
    fi

    $jobruncmd -q -n $procs $npstr $hstr $envstr $bind_opt $extra_opts \
      ${DEFAULT_MPIOPTS-} $exe 2>&1 |
      tee -a $log2 | tee -a $log1

  elif [ $jobruncli = srun ]; then

    # we are running slurm....
    if [ ${#envs[@]} -gt 0 ]; then
      envstr=$(printf -- "%s=%s," ${envs[@]})
      # XXX: "ALL" isn't documented in srun(1) man page, but it works.
      # without it, all other env vars are removed (e.g. as described
      # in the sbatch(1) man page ...).
      envstr="--export=${envstr}ALL"
    fi

    if [ $ppnode -gt 0 ]; then
      nnodes=$((procs / ppnode))
      npstr="-N $nnodes --ntasks-per-node=$ppnode"
    fi

    if [ ! -z "$hosts" ]; then
      hstr="-w $hosts"
    fi

    if [ x$bind_opt = xthread ]; then
      bind_opt="--cpu-bind=threads"
    elif [ x$bind_opt = xcore ]; then
      bind_opt="--cpu-bind=cores"
    elif [ x$bind_opt = xexternal ]; then
      bind_opt=""
    elif [ x$bind_opt = xnone -o x$bind_opt = x ]; then
      bind_opt="--cpu-bind=none"
    else
      die "do_mpirun: bad bind ops $bind_opt"
    fi

    message "[MPIEXEC]" "$jobruncmd -n $procs" $npstr $hstr $envstr \
      $bind_opt $extra_opts ${DEFAULT_MPIOPTS-} $exe

    if [ x${DRYRUN-0} != x0 ]; then
      return
    fi

    $jobruncmd -n $procs $npstr $hstr $envstr $bind_opt $extra_opts \
      ${DEFAULT_MPIOPTS-} $exe 2>&1 |
      tee -a $log2 | tee -a $log1

  elif [ $jobruncli = mpich ]; then

    if [ ${#envs[@]} -gt 0 ]; then
      envstr=$(printf -- "-env %s %s " ${envs[@]})
    fi

    if [ $ppnode -gt 0 ]; then
      npstr="-ppn $ppnode"
    fi

    if [ ! -z "$hosts" ]; then
      hstr="--host $hosts"
    fi

    if [ x$bind_opt = xthread ]; then
      bind_opt="-bind-to=hwthread"
    elif [ x$bind_opt = xcore ]; then
      bind_opt="-bind-to=core"
    elif [ x$bind_opt = xexternal ]; then
      bind_opt=""
    elif [ x$bind_opt = xnone -o x$bind_opt = x ]; then
      bind_opt="-bind-to=none"
    else
      die "do_mpirun: bad bind ops $bind_opt"
    fi

    message "[MPIEXEC]" "$jobruncmd -np $procs" $npstr $hstr $envstr \
      $bind_opt $extra_opts ${DEFAULT_MPIOPTS-} $exe

    if [ x${DRYRUN-0} != x0 ]; then
      return
    fi

    $jobruncmd -np $procs $npstr $hstr $envstr $bind_opt $extra_opts \
      ${DEFAULT_MPIOPTS-} $exe 2>&1 |
      tee -a $log2 | tee -a $log1

  elif [ $jobruncli = openmpi ]; then

    if [ ${#envs[@]} -gt 0 ]; then
      envstr=$(printf -- "-x %s=%s " ${envs[@]})
    fi

    if [ $ppnode -gt 0 ]; then
      npstr="-npernode $ppnode"
    fi

    if [ ! -z "$hosts" ]; then
      if [ $ppnode -gt 1 ]; then
        hhstr="$(printf '&,%.0s' $(seq 1 $(($ppnode - 1))))"
        hhstr="$(echo $hosts | sed -e 's/\([^,]*\)/'"$hhstr&"'/g')"
        hstr="--host $hhstr"
      else
        hstr="--host $hosts"
      fi
    fi

    if [ x$openmpi_newbind = x1 ]; then
      if [ x$bind_opt = xthread ]; then
        bind_opt="--bind-to hwthread"
      elif [ x$bind_opt = xcore ]; then
        bind_opt="--bind-to core"
      elif [ x$bind_opt = xexternal ]; then
        bind_opt=""
      elif [ x$bind_opt = xnone -o x$bind_opt = x ]; then
        bind_opt="--bind-to none"
      else
        die "do_mpirun: bad bind ops $bind_opt"
      fi
    else
      if [ x$bind_opt = xthread ]; then
        message "do_mpirun: aprun: converted thread CPU binding to core"
        bind_opt="-bind-to-core"
      elif [ x$bind_opt = xcore ]; then
        bind_opt="-bind-to-core"
      elif [ x$bind_opt = xexternal ]; then
        bind_opt=""
      elif [ x$bind_opt = xnone -o x$bind_opt = x ]; then
        bind_opt="-bind-to-none"
      else
        die "do_mpirun: bad bind ops $bind_opt"
      fi
    fi

    message "[MPIEXEC]" "$jobruncmd -n $procs" $npstr $hstr $envstr \
      $bind_opt $extra_opts ${DEFAULT_MPIOPTS-} $exe

    if [ x${DRYRUN-0} != x0 ]; then
      return
    fi

    $jobruncmd -n $procs $npstr $hstr $envstr $bind_opt $extra_opts \
      ${DEFAULT_MPIOPTS-} $exe 2>&1 |
      tee -a $log2 | tee -a $log1

  else
    die "could not find a supported do_mpirun command"
  fi
}

#
# call common_init unless its been disabled (e.g. for debugging)
#
if [ x${common_noinit-} = x ]; then
    common_init
fi
