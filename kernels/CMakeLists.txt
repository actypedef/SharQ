cmake_minimum_required(VERSION 3.18)
project(SharQ LANGUAGES CXX CUDA)

set(KERNELS_DIR ${CMAKE_CURRENT_LIST_DIR})
set(KERNELS_SRC_DIR ${KERNELS_DIR}/src)
set(KERNELS_INCLUDE_DIR ${KERNELS_DIR}/include)
set(KERNELS_BENCH_DIR ${KERNELS_DIR}/benchmark)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_found
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
  OUTPUT_VARIABLE TORCH_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_pkg_found
)

if(NOT _torch_found EQUAL 0)
  message(FATAL_ERROR "PyTorch not found. Please make sure 'import torch' works in your Python environment.")
else()
  list(INSERT CMAKE_PREFIX_PATH 0 "${TORCH_CMAKE_PREFIX_PATH}")
  message(STATUS "Found PyTorch CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")
endif()

if(NOT _torch_pkg_found EQUAL 0)
  message(FATAL_ERROR "Failed to locate the installed torch package directory from Python3_EXECUTABLE.")
else()
  message(STATUS "Found PyTorch package dir: ${TORCH_PACKAGE_DIR}")
endif()

if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
  if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
  else()
    set(TORCH_CUDA_ARCH_LIST "12.0a")
    set(ENV{TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST}")
  endif()
endif()
message(STATUS "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")

unset(CMAKE_CUDA_ARCHITECTURES)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)

find_package(Torch REQUIRED)

find_package(CUDAToolkit REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
find_package(pybind11 CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_PACKAGE_DIR}/lib" NO_DEFAULT_PATH)

set(CUTLASS_ROOT ${KERNELS_DIR}/../third-party/cutlass)

include_directories(${KERNELS_INCLUDE_DIR})
include_directories(/usr/local/cuda/include)
link_directories("${TORCH_PACKAGE_DIR}/lib")
include_directories("${TORCH_PACKAGE_DIR}/include")
include_directories("${TORCH_PACKAGE_DIR}/include/torch/csrc/api/include")
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/tools/util/include)
include_directories(${CUTLASS_ROOT}/examples/common)

add_compile_options(-w)
add_compile_options(-fpermissive)

function(enable_cuda_sm120a target_name)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES OFF)
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_120a,code=sm_120a>)
endfunction()

function(use_torch_cuda_arch_flags target_name)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES OFF)
endfunction()

add_library(gemm_dense_objs OBJECT
  ${KERNELS_SRC_DIR}/nvfp4.cu
  ${KERNELS_SRC_DIR}/reorder.cu
  ${KERNELS_SRC_DIR}/down.cu)
add_library(gemm_sparse_objs OBJECT
  ${KERNELS_SRC_DIR}/sparse_nvfp4.cu
  ${KERNELS_SRC_DIR}/fused_sparse_prepare.cu
  ${KERNELS_SRC_DIR}/shared_weight_nvfp4.cu)


add_executable(bench_nvfp4 ${KERNELS_BENCH_DIR}/bench_nvfp4.cu
$<TARGET_OBJECTS:gemm_dense_objs>)

add_executable(bench_sparse_nvfp4 ${KERNELS_BENCH_DIR}/bench_sparse_nvfp4.cu
$<TARGET_OBJECTS:gemm_dense_objs>
$<TARGET_OBJECTS:gemm_sparse_objs>)


add_library(sharq_ops MODULE ${KERNELS_SRC_DIR}/bindings.cpp
  $<TARGET_OBJECTS:gemm_dense_objs>
  $<TARGET_OBJECTS:gemm_sparse_objs>)
set_source_files_properties(${KERNELS_SRC_DIR}/bindings.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(sharq_ops PRIVATE pybind11::module "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart)

target_compile_options(bench_nvfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_sparse_nvfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(gemm_dense_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
target_compile_options(gemm_sparse_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
target_compile_options(sharq_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

use_torch_cuda_arch_flags(gemm_dense_objs)
use_torch_cuda_arch_flags(bench_nvfp4)
use_torch_cuda_arch_flags(sharq_ops)
enable_cuda_sm120a(gemm_sparse_objs)
enable_cuda_sm120a(bench_sparse_nvfp4)

target_compile_definitions(bench_nvfp4 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_sparse_nvfp4 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(sharq_ops PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)

set_property(TARGET sharq_ops PROPERTY CXX_STANDARD 17)
set_target_properties(sharq_ops PROPERTIES PREFIX "")
add_definitions(-DTORCH_EXTENSION_NAME=sharq_ops)



