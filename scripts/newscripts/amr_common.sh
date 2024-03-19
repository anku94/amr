#!/usr/bin/env bash

set -eu

#
# amr_prepare_expdir: prepare variables for a single experiment
# and create the expdir. must be done before amr_prepare_deck
#
# uses: amr_bin, lb_policy, cores, nodes
# sets: exp_tag, exp_jobdir, exp_logfile
# creates: exp_jobdir
#
amr_prepare_expdir() {
	local bin_name=$(basename $amr_bin)
	exp_tag="${bin_name}_LB${lb_policy}_C${cores}_N${nodes}"

	message "-INFO- Will use exp dir: $exp_tag in jobdir"
	exp_jobdir="$jobdir/$exp_tag"
	exp_logfile="$exp_jobdir/$exp_tag.log"

	mkdir -p $exp_jobdir
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
	local deck_out_fpath="$exp_jobdir/${deck_name}"

	[ -f "$deck_in_fpath" ] || die "!! ERROR !! Deck file not found: $deck_in_fpath"

	message "-INFO- Preparing deck from template: $deck_in_fpath"
	message "-INFO- Generated deck will be: $deck_out_fpath"

	cat $deck_in_fpath |
		sed -s "s/{LB_POLICY}/$deck_lb_policy/g" |
		sed -s "s/{NLIM}/$deck_nlim/g" >$deck_out_fpath

	message "-INFO- Deck prepared"

}

#
# amr_do_run: run an AMR experiment
# uses: $amru_prefix, $jobdir, $cores, $nodes, $ppn, $logfile, $exp_logfile,
# $amr_nodes, $amr_cpubind
#
# creates:
# side effects: changes current directory to $exp_jobdir
#
# Arguments:

amr_do_run() {
	prelib="${1}"
	amr_bin="${2}"
	amr_deck="${3}"

	local amr_bin_fullpath="$amru_prefix/bin/$amr_bin"
	local amr_deck_fullpath="$exp_jobdir/$amr_deck"
	local amr_exec="$amr_bin_fullpath -i $amr_deck_fullpath"

	[ -f "$amr_bin_fullpath" ] || die "AMR binary not found: $amr_bin_fullpath"
	[ -f "$amr_deck_fullpath" ] || die "AMR deck not found: $amr_deck_fullpath"

	message "-INFO- Running AMR experiment: $amr_exec"

	prelib_env=()
	if [ x"$prelib" != x ]; then
		# If prelib is a relative path, check for its existence in lib/
		if [ "${prelib:0:1}" != "/" ]; then
			prelib="$amru_prefix/lib/$prelib"
			[ -f "$prelib" ] || die "Preload lib not found: $prelib"
		fi

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

	do_mpirun $cores $ppn "$amr_cpubind" env_vars[@] \
		"$amr_nodes" "$amr_exec" "${EXTRA_MPIOPTS-}"
}
