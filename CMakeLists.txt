cmake_minimum_required(VERSION 3.15)
project(matrix_dot_mpi C)

set(CMAKE_C_STANDARD 99)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES matrix_dot_mpi.c)

add_executable(matrix_dot_mpi matrix_dot_mpi.c)
