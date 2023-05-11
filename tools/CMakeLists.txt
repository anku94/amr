set (common_include "${CMAKE_SOURCE_DIR}/tools/common/")
set (common_srcs "${CMAKE_SOURCE_DIR}/tools/common/common.cc")

# BEGIN GTest
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
# END GTest

option(FORT_ENABLE_TESTING OFF)
option(FORT_TEST_BUILD OFF)
option(INSTALL_GTEST OFF)

set(BUILD_SHARED_LIBS OFF)
add_subdirectory(external/libfort)
set_target_properties(fort PROPERTIES EXCLUDE_FROM_ALL TRUE)
set(BUILD_SHARED_LIBS ON)

add_subdirectory(topobench)
add_subdirectory(loadbalancing)
