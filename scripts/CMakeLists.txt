set (install-programs-in
    newscripts/amr-minimal.sh.in newscripts/run_amr_test.sh.in)

set (install-programs
  newscripts/amr_common.sh)

foreach (lcv ${install-programs-in})
  string (REGEX REPLACE "\\.in$" "" out "${lcv}")
  configure_file (${lcv} ${out} @ONLY)
  list (APPEND install-programs-in-done ${CMAKE_CURRENT_BINARY_DIR}/${out})
endforeach ()

install (PROGRAMS ${install-programs} DESTINATION scripts)
install (PROGRAMS ${install-programs-in-done} DESTINATION scripts)
