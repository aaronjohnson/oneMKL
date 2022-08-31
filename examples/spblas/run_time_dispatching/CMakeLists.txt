#===============================================================================
# Copyright 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

# NOTE: user needs to set env var SYCL_DEVICE_FILTER to use runtime example without specifying backend in CMake
# $ENV{SYCL_DEVICE_FILTER}

# Build object from all example sources
set(SPBLAS_RT_SOURCES "gemv")

# Set up for the right backend for run-time dispatching examples
# If users build more than one backend (i.e. mklcpu and mklgpu, or mklcpu and CUDA), they may need to
# overwrite SYCL_DEVICE_FILTER in their environment to run on the desired backend
set(DEVICE_FILTERS "")
if(ENABLE_MKLCPU_BACKEND)
  list(APPEND DEVICE_FILTERS "cpu")
endif()
if(ENABLE_MKLGPU_BACKEND)
  list(APPEND DEVICE_FILTERS "gpu")
endif()
if(ENABLE_CUSPBLAS_BACKEND)
  list(APPEND DEVICE_FILTERS "cuda:gpu")
endif()
if(ENABLE_ROCSPBLAS_BACKEND)
  list(APPEND DEVICE_FILTERS "hip:gpu")
endif()

message(STATUS "SYCL_DEVICE_FILTER will be set to the following value(s): [${DEVICE_FILTERS}] for run-time dispatching examples")

foreach(spblas_rt_source ${SPBLAS_RT_SOURCES})
  set(EXECUTABLE example_rt_${domain}_${spblas_rt_source})
  add_executable(${EXECUTABLE} ${spblas_rt_source}.cpp)
  target_include_directories(${EXECUTABLE}
      PUBLIC ${PROJECT_SOURCE_DIR}/examples/include
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PUBLIC ${CMAKE_BINARY_DIR}/bin
  )

  add_dependencies(${EXECUTABLE} onemkl)

  if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${SPBLAS_RT_SOURCES})
  endif()

  target_link_libraries(${EXECUTABLE} PUBLIC
      onemkl
      ONEMKL::SYCL::SYCL
      ${CMAKE_DL_LIBS}
  )

  # Register example as ctest
  foreach(device_filter ${DEVICE_FILTERS})
    set(ID example_${domain}_${spblas_rt_source}/${device_filter})
    add_test(NAME ${ID} COMMAND ${EXECUTABLE})
    set_property(TEST ${ID} PROPERTY
      ENVIRONMENT LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
      ENVIRONMENT SYCL_DEVICE_FILTER=${device_filter})
  endforeach(device_filter)

endforeach(spblas_rt_source)
