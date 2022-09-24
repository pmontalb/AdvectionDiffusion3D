
include_guard()

cmake_minimum_required(VERSION 3.14)
project(pde)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)

add_subdirectory(Pde)
add_subdirectory(UnitTests)
