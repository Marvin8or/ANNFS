Cmake_minimum_required(VERSION 3.8)
project(ANNFS VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(Utils)

add_executable(ANNFS main.cpp)

target_include_directories(ANNFS PRIVATE "${PROJECT_SOURCE_DIR}/Math/LinearAlgebra/Matrix.h")
target_link_libraries(ANNFS PUBLIC Utils)

target_include_directories(ANNFS PUBLIC
                       	  "${PROJECT_BINARY_DIR}"
			  "${PROJECT_SOURCE_DIR}/Utils"
                          )