set (install-programs-in
    amr-minimal.sh.in 
    run_amr_test.sh.in
    amr_job_template.txt.in
    run_topo_test.sh.in
    topo_job_template.txt.in
  )

set (install-programs
  amr_common.sh
  amr_suite.csv
  gen_amr_jobs.py
  gen_topo_jobs.py
)

foreach (lcv ${install-programs-in})
  string (REGEX REPLACE "\\.in$" "" out "${lcv}")
  configure_file (${lcv} ${out} @ONLY)
  list (APPEND install-programs-in-done ${CMAKE_CURRENT_BINARY_DIR}/${out})
endforeach ()

install (PROGRAMS ${install-programs} DESTINATION scripts)
install (PROGRAMS ${install-programs-in-done} DESTINATION scripts)
