
include_guard()

cmake_minimum_required(VERSION 3.14)
project(pde)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(MKL REQUIRED)
#add_compile_definitions(EIGEN_USE_MKL_ALL)

#find_package(OpenMP REQUIRED)

add_subdirectory(Discretizers)
add_subdirectory(Pde)
add_subdirectory(UnitTests)
add_subdirectory(Profiler)
