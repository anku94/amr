#!/usr/bin/env bash
set -u

# Wrapper script to invoke a MPI binary with perf record and control fd's.
# Sets up CTL and ACK fds, and passes them to the program via
# env vars PERF_CTL_FD and PERF_ACK_FD. Temporary FIFOs at $FIFO_PREFIX.
#
# Written to be used with MVAPICH2, other MPIs not tested.
#
# Uses:
# - PERF_OUTPUT_DIR: Dir to store perf.data.<rank> files
# - PMI_RANK: Assumed to be the rank of the process, set by hydra

FIFO_PREFIX=/tmp/perfctl

setup_paths() {
  local -i FDRANK=$1

  FIFO_CTL=${FIFO_PREFIX}.ctl.$FDRANK
  FIFO_ACK=${FIFO_PREFIX}.ack.$FDRANK
  PERF_OUTPUT_DIR=${PERF_OUTPUT_DIR:-/tmp}
  PERF_OUTPUT_FILE=$PERF_OUTPUT_DIR/perf.data.$FDRANK
}

setup_ctlfds() {
  local -i FDRANK=$1

  mkfifo $FIFO_CTL
  mkfifo $FIFO_ACK

  exec {perf_ctl_fd}<>$FIFO_CTL
  exec {perf_ack_fd}<>$FIFO_ACK

  export PERF_CTL_FD=${perf_ctl_fd}
  export PERF_ACK_FD=${perf_ack_fd}
}

cleanup_ctlfds() {
  local -i FDRANK=$1

  # clean up fd's and delete the fifo's
  exec {perf_ctl_fd}>&-
  exec {perf_ack_fd}>&-

  rm -f $FIFO_CTL
  rm -f $FIFO_ACK
}

run_cmd() {
  perf record -o $PERF_OUTPUT_FILE \
    --call-graph dwarf \
    --control fd:${perf_ctl_fd},${perf_ack_fd} \
    --delay=-1 \
    "$@"
}

run_on_mvapich() {
  setup_paths $PMI_RANK
  setup_ctlfds $PMI_RANK
  run_cmd "$@"
  cleanup_ctlfds $PMI_RANK
}

if [ $# -eq 0 ]; then
  echo "Usage: $0 <cmd> [args...]"
  exit 1
fi

run_on_mvapich "$@"
