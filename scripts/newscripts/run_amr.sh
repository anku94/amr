#!/usr/bin/env bash
#
#
arg_test_type="baseline"
arg_host_suffix="-dib"
arg_ip_subnet="10.94"
arg_amr_bin="advection-example"
arg_amr_deck="advection-example.deck"
arg_pre="${AMR_PRE-libprof-preload.so}"
arg_nlim="${AMR_NLIM-1}"
arg_lb_policy="${AMR_LB_POLICY-baseline}"

echo "Running AMR test with the following parameters:"
echo "  Test type: $arg_test_type"
echo "  Host suffix: $arg_host_suffix"
echo "  IP subnet: $arg_ip_subnet"
echo "  AMR binary: $arg_amr_bin"
echo "  AMR deck: $arg_amr_deck"
echo "  AMR preload: $arg_pre"
echo "  AMR nlim: $arg_nlim"
echo "  AMR lb policy: $arg_lb_policy"

prelib="ld-lol.so"

message() {
	echo $1
}

die() {
	message $1
	exit -1
}

#
# amr_prepare_expdir
#
amr_prepare_expdir() {
	expdir="binname_Ppolicy_C512_N16"
}

#
# amr_prepare_deck: prepare the AMR deck
#
# uses:
# creates:

amr_prepare_deck() {
	deck_name=${1}
	deck_lb_policy=${2}
	deck_nlim=${3}

	local deck_in_fpath="$amru_prefix/decks/${deck_name}.in"
	[ -f "$deck_in_fpath" ] || die "!! ERROR !! Deck file not found: $deck_in_fpath"

	cat $deck_in_fpath |
		sed -s "s/{LB_POLICY}/$deck_lb_policy/g" |
		sed -s "s/{NLIM}/$deck_nlim/g" >$deck_in_fpath

	echo "Looking for "

}

#
# amr_do_run: run an AMR experiment
# uses:
# creates:
# side effects:
#
# Arguments:

amr_do_run() {
	prelib_env=()
	if [ x"$prelib" != x ]; then
		prelib_env=(
			"LD_PRELOAD" "$prelib"
			"KOKKOS_TOOLS_LIBS" "$prelib"
		)
	fi

	# create an array called env_vars, and insert prelib_env into that array
	env_vars=(
		"${prelib_env[@]}"
		"GLOG_minloglevel" ${GLOG_minloglevel-0}
		"GLOG_v" ${GLOG_v-0}
	)

	# iterate thru env_vars and print everything
	for i in "${env_vars[@]}"; do
		echo $i
	done
}

amr_do_run
