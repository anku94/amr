#!/bin/bash -eu

. mpi_common.sh

add_env_var() {
  if [[ "$MPITYPE" == "openmpi" ]]; then
    env_flag="-x"
    env_sep="="
  else
    env_flag="-env"
    env_sep=" "
  fi

  # ENV_STR="$ENV_STR $env_flag ${1}${env_sep}${2}"
  export $1=$2
  ENV_STR="$1 $2"
}


binding_flags() {
  echo "Error: ENV_STR not modified properly. Check code, remove this warning."
  exit 1

  ENV_STR="--bind-to none"

  add_env_var OMP_DISPLAY_ENV true
  add_env_var OMP_NUM_THREADS 16

  # ENV_STR="$ENV_STR -x OMP_DISPLAY_ENV=true"
  # ENV_STR="$ENV_STR -x OMP_NUM_THREADS=16"
  # ENV_STR="$ENV_STR -x OMP_PLACES=threads"
  # ENV_STR="$ENV_STR -x OMP_PROC_BIND=spread"
  echo $ENV_STR
}

mca_flags() {
  echo --mca btl tcp,self,vader --mca btl_tcp_if_include ibs2 --mca pml ob1
}

confirm_assume_yes() {
  read -p "Continue? (Y/n): " confirm
  if [[ $confirm == [nN] || $confirm == [nN][oO] ]]; then
    exit 1
  fi
}

confirm_assume_no() {
  read -p "Continue? (y/N): " confirm
  if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    exit 1
  fi
}

# XXX: unused
gen_hostfile_rr() {
  nnodes=32
  nrpn=16

  ncontig=2
  nrr=$(( nrpn / ncontig ))

  rm hosts.rr.txt

  for rr_ridx in $( seq $nrr );
  do
    for hidx in $( seq 0 $(( nnodes - 1 )) ); do
      echo h${hidx}-dib:$ncontig >> hosts.rr.txt
    done
  done
}

#
# argument: all nodes to add to blacklist, space sparated
# 

update_blacklist() {
  blacklist=$1

  if [[ "${exp_hosts_blacklist:-"none"}" == "none" ]]; then
    message "-INFO- No blacklist file setup!!!"
    return
  fi

  for node in $blacklist; do

    if [[ "$node" == h* ]]; then
      node_alias=$node
    else
      node_alias=$( ssh $node "hostname | cut -d. -f 1" )
    fi

    host_suffix_tmp=${host_suffix:-"none"}
    if [[ "$host_suffix_tmp" != "none" ]]; then
      node_alias=$node_alias-$host_suffix_tmp
    fi

    message "-INFO- Blacklisting $node_alias"
    echo $node_alias'([^0-9]|$)' >> $exp_hosts_blacklist
  done
}

log_throttling_nodes() {
  THROTTLE_SCRIPT=/users/ankushj/snips/scripts/log-throttlers.sh

  nnodes=$(cat $jobdir/hosts.txt | wc -l)
  all_nodes=$(cat $jobdir/hosts.txt | paste -sd,)
  CHECK=$(do_mpirun $nnodes 1 "none" "" "$all_nodes" $THROTTLE_SCRIPT | tail -n+2 | cut -d, -f1)

  echo $CHECK
}
