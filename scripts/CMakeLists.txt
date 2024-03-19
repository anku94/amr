# set (install-scripts-in )
set (install-scripts 
  newscripts/amr_common.sh
  newscripts/mpi_common.sh
  newscripts/run_amr.sh
  newscripts/run_amr_test.sh)

# install (FILES ${install-scripts-in} DESTINATION scripts)
install (FILES ${install-scripts} DESTINATION scripts)
