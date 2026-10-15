cmake_minimum_required(VERSION 3.21)
project(distributed_kernels LANGUAGES CXX)

# ---------------------------
# Basics
# ---------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Repo root is always ../ relative to this CMakeLists.txt
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
message(STATUS "REPO_ROOT: ${REPO_ROOT}")

# ThunderKittens root == repo root
set(THUNDERKITTENS_ROOT "${REPO_ROOT}")
if(NOT EXISTS "${THUNDERKITTENS_ROOT}/include")
  message(FATAL_ERROR "Bad THUNDERKITTENS_ROOT (${THUNDERKITTENS_ROOT}): missing include/")
endif()

# ---------------------------
# ROCm / hipcc
# ---------------------------
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm install path")
if(NOT ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install path" FORCE)
endif()

# NOTE: best used with a fresh build dir; if you previously configured with a different compiler,
# remove build/ and reconfigure.
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc" CACHE FILEPATH "C++ compiler" FORCE)

message(STATUS "ROCM_PATH: ${ROCM_PATH}")
message(STATUS "Using hipcc: ${CMAKE_CXX_COMPILER}")

# ---------------------------
# GPU target flags
# ---------------------------
set(GPU_TARGET "CDNA4" CACHE STRING "GPU target: CDNA2|CDNA3|CDNA4|RDNA3")

if(GPU_TARGET STREQUAL "CDNA2")
  set(GPU_FLAGS "-DKITTENS_CDNA2" "--offload-arch=gfx90a")
elseif(GPU_TARGET STREQUAL "CDNA3")
  set(GPU_FLAGS "-DKITTENS_CDNA3" "--offload-arch=gfx942")
elseif(GPU_TARGET STREQUAL "CDNA4")
  set(GPU_FLAGS "-DKITTENS_CDNA4" "--offload-arch=gfx950" "-DHIP_ENABLE_WARP_SYNC_BUILTINS" "-ffast-math")
elseif(GPU_TARGET STREQUAL "RDNA3")
  set(GPU_FLAGS "-DKITTENS_RDNA3" "--offload-arch=gfx1100")
else()
  message(FATAL_ERROR "Unknown GPU_TARGET: ${GPU_TARGET}")
endif()

message(STATUS "GPU_TARGET: ${GPU_TARGET}")

# ---------------------------
# CPM.cmake (download if needed)
# ---------------------------
set(CPM_DOWNLOAD_VERSION 0.38.7)
if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION}
  )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# ---------------------------
# Iris (disable extras, fetch via CPM)
# ---------------------------
set(IRIS_BUILD_EXAMPLES OFF CACHE BOOL "Build iris examples" FORCE)
set(IRIS_BUILD_BENCHMARKS OFF CACHE BOOL "Build iris benchmarks" FORCE)
set(IRIS_BUILD_TESTS OFF CACHE BOOL "Build iris tests" FORCE)

# Offline builds: point IRIS_SOURCE_DIR at a checkout of ROCm/iris and CPM will
# use it instead of reaching for GitHub. Needed on air-gapped nodes, which is
# also where this is most likely to be run.
set(IRIS_SOURCE_DIR "" CACHE PATH "Local checkout of ROCm/iris (skips the download)")
if(IRIS_SOURCE_DIR)
  if(NOT EXISTS "${IRIS_SOURCE_DIR}/irisx/CMakeLists.txt")
    message(FATAL_ERROR "IRIS_SOURCE_DIR=${IRIS_SOURCE_DIR} has no irisx/CMakeLists.txt")
  endif()
  message(STATUS "Using local iris: ${IRIS_SOURCE_DIR}")
  set(CPM_iris_SOURCE "${IRIS_SOURCE_DIR}")
endif()

CPMAddPackage(
  NAME iris
  GITHUB_REPOSITORY ROCm/iris
  GIT_TAG muhaawd/irisx
  SOURCE_SUBDIR irisx
  OPTIONS
    "IRIS_BUILD_EXAMPLES OFF"
    "IRIS_BUILD_BENCHMARKS OFF"
    "IRIS_BUILD_TESTS OFF"
)

# ---------------------------
# MPI + Python + pybind11
# ---------------------------
find_package(MPI REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# ---------------------------
# Common interface target for all modules
# ---------------------------
add_library(dk_common INTERFACE)
target_compile_options(dk_common INTERFACE
  ${GPU_FLAGS}
  -w
  -Rpass-analysis=kernel-resource-usage
)
target_include_directories(dk_common INTERFACE
  "${THUNDERKITTENS_ROOT}/include"
  "${THUNDERKITTENS_ROOT}/prototype"
  "${ROCM_PATH}/include/hip"
)
target_link_libraries(dk_common INTERFACE
  iris::iris
  ${Python3_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  -ldl
  -lm
)

# ---------------------------
# Build selection: ALL or one kernel dir name
# ---------------------------
set(DK_BUILD "ALL" CACHE STRING "Which kernel to build: ALL or subdir name (e.g. bf16_gemm)")

# Auto-discover kernel subdirs: any directory containing kernel.cpp
file(GLOB children RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*")
set(DK_KERNEL_DIRS "")
foreach(child ${children})
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${child}" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${child}/kernel.cpp")
    list(APPEND DK_KERNEL_DIRS "${child}")
  endif()
endforeach()

if(NOT DK_KERNEL_DIRS)
  message(FATAL_ERROR "No kernel subdirs found (expected */kernel.cpp under ${CMAKE_CURRENT_SOURCE_DIR})")
endif()

message(STATUS "Discovered kernels: ${DK_KERNEL_DIRS}")

# Each kernel builds as a UNIQUE CMake target (dir name), but outputs a Python module named tk_kernel
# into that kernel's directory.
function(add_dk_kernel dir)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/kernel.cpp")
  set(target "${dir}")  # unique CMake target name

  pybind11_add_module(${target} "${src}")
  target_link_libraries(${target} PRIVATE dk_common)

  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tk_kernel"                    # <-- always import as tk_kernel
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}"  # next to example.py
  )

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Built: ${dir}/tk_kernel.${Python3_SOABI}.so"
  )
endfunction()

if(DK_BUILD STREQUAL "ALL")
  foreach(d ${DK_KERNEL_DIRS})
    add_dk_kernel(${d})
  endforeach()
else()
  list(FIND DK_KERNEL_DIRS "${DK_BUILD}" idx)
  if(idx EQUAL -1)
    message(FATAL_ERROR "DK_BUILD='${DK_BUILD}' not found. Available: ${DK_KERNEL_DIRS}")
  endif()
  add_dk_kernel(${DK_BUILD})
endif()

# ---------------------------
# iris_py module (shared infra) -> output into distributed-kernels/
# ---------------------------
pybind11_add_module(iris_py "${CMAKE_CURRENT_SOURCE_DIR}/iris_py.cpp")
target_link_libraries(iris_py PRIVATE dk_common)
set_target_properties(iris_py PROPERTIES
  PREFIX ""
  OUTPUT_NAME "iris_py"
  SUFFIX ".${Python3_SOABI}.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_command(TARGET iris_py POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Built: iris_py.${Python3_SOABI}.so"
)
