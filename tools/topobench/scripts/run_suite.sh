#!/usr/bin/env bash

set -eu

dims_xyz=2
max_reflvl=4
target_leafcnt=16
all_policies=( "baseline" "cdp" "hybrid25" "hybrid50" "hybrid75" "lpt" )
#all_policies=( "hybrid25" )
jobdir=/tmp
all_num_ts=( 1 10 100 )
all_num_ts=( 1 )
all_num_rounds=( 1 10 100 )
all_num_rounds=( 1 )

# TODO: set AMRLB_CONFIG and AMRLB_LOGDIR
TOPOBENCH_BIN=/Users/schwifty/Repos/amr/build/tools/topobench/src/meshdriver_main

# getdatetime: returns YYYYMMDD_HHMMSS
getdatetime() {
    date +%Y%m%d_%H%M%S
}

getlogfname() {
    # mrl: max_revlvl
    # tlc: target_leafcnt
    echo "topobench_mesh_$(getdatetime)_dims${dims_xyz}_mrl${max_reflvl}_tlc${target_leafcnt}.csv"
}

message() {
    echo "[INFO] $@"
}

# gen_config: assumes all vars are set
gen_config() {
    cfg_template=$1
    cfg_out=$2

    [ ! -f $cfg_template ] && {
        message "Config template $cfg_template does not exist"
        exit 1
    }

    [ -f $cfg_out ] && {
        message "Config output $cfg_out already exists"
        message "Removing it.."
        rm -f $cfg_out
    }

    sed -e "s|@dims_xyz@|$dims_xyz|" \
        -e "s|@max_reflvl@|$max_reflvl|" \
        -e "s|@target_leafcnt@|$target_leafcnt|" \
        -e "s|@policy@|$policy|" \
        -e "s|@jobdir@|$jobdir|" \
        -e "s|@log_fname@|$log_fname|" \
        -e "s|@num_ts@|$num_ts|" \
        -e "s|@num_rounds@|$num_rounds|" \
        $cfg_template > $cfg_out
}

gen_cfg_test() {
    cfg_template=config/template.cfg
    cfg_out=config/cfg-test.cfg
    gen_config $cfg_template $cfg_out
}

run_single() {
    message "Running with policy $policy"

    cfg_template=config/template.cfg
    cfg_out=config/cfg-test.cfg
    gen_config $cfg_template $cfg_out

    export AMRLB_CONFIG=$cfg_out
    export AMRLB_LOGDIR=$jobdir
    GLOG_v=-1 mpirun -np 8 $TOPOBENCH_BIN
}

run() {
    log_fname=$(getlogfname)

    message "Log file: $log_fname"
    # gen_cfg_test
    for policy in ${all_policies[@]}; do
        for num_ts in ${all_num_ts[@]}; do
            for num_rounds in ${all_num_rounds[@]}; do
                message "Running with policy $policy, num_ts $num_ts, num_rounds $num_rounds"
                # run_single
            done
        done
    done
}

run