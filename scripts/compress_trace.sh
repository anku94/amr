#!/usr/bin/env bash

# Usage Instructions: set nworkers=8 (for 8 nodes)
# Run on h0-h7. Each of them will get a unique shard of the work
# If you run on h0 8 times, it will also complete the compression task.
# As the shard is defined on "remaining work"

set -u

get_files() {
  trace_dir=$1
  widx=$2
  wtotal=$3

  ntasks=$(fd -t f $search_dir $1 | wc -l)
  ntasks_pw=$(( ntasks / wtotal ))
  wt_beg=$(( widx * ntasks_pw ))
  wt_end=$(( wt_beg + ntasks_pw ))

  # echo "Processing tasks between: $wt_beg, $wt_end"

  echo $(fd -t f $search_dir $1 | grep -v gz | head -$wt_end | tail -$ntasks_pw)
}

get_hidx() {
  echo $(hostname) | cut -d. -f 1 | sed 's/h//g'
}

run_for_pattern() {
  get_files $trace_dir $hidx $nworkers | sed 's/\ /\n/g' | parallel -I% --max-args 1 gzip %
}

run() {
  trace_dir=$1
  nworkers=$2

  hidx=$(get_hidx)

  if [[ $hidx -ge $nworkers ]]; then
    echo "Don't need to do anything"
    exit 0
  fi

  # search_dir=msgs
  run_for_pattern
}

if [ $# != 2 ]; then
  echo "Insufficient arguments. Run: $0 \$trace_dir \$search_dir"
  exit 1
fi

nworkers=8

trace_dir=$1
search_dir=$2

run $trace_dir $nworkers
