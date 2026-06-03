cmake_minimum_required(VERSION 3.15)
project(SNSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_snsolver
    src/quadrature.cpp
    src/mesh.cpp
    src/xsdata.cpp
    src/dsa.cpp
    src/kinetics.cpp
    src/quasistatic.cpp
    src/solver.cpp
    src/bindings.cpp
)

target_include_directories(_snsolver PRIVATE ${CMAKE_SOURCE_DIR}/src)
