cmake_minimum_required(VERSION 3.10)
project(bench)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add executable
add_executable(${PROJECT_NAME} main.cc)