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
include(GoogleTest)

set(amr-tests topogen_test.cc)

foreach (lcv ${amr-tests})
    get_filename_component(id ${lcv} NAME_WE)
    message(STATUS "Building test: ${id}")
    add_executable(${id} EXCLUDE_FROM_ALL ${lcv})
    target_link_libraries(${id} GTest::gtest_main topo)
    gtest_discover_tests(${id})
endforeach()