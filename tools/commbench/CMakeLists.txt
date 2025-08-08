add_subdirectory(src)
add_subdirectory(tests)

# copy all files matching *.sh in scripts to install prefix/scripts use
# FILES_MATCHING_PATTERN to match *.sh
install(
  DIRECTORY config/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/config
  FILES_MATCHING
  PATTERN "*.cfg")

install(
  DIRECTORY scripts/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/scripts
  FILES_MATCHING
  PATTERN "*.sh")
