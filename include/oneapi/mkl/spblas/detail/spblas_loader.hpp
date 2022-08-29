/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _ONEMKL_SPBLAS_LOADER_HPP_
#define _ONEMKL_SPBLAS_LOADER_HPP_

#include <complex>
#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi {
namespace mkl {
namespace spblas {
namespace detail {

// Buffer APIs

ONEMKL_EXPORT void gemv(oneapi::mkl::device                  libkey, 
                        sycl::queue                          &queue,
                        oneapi::mkl::transpose               transpose_val,
                        const fp                             alpha,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        sycl::buffer<fp, 1>                  &x,
                        const fp                             beta,
                        sycl::buffer<fp, 1>                  &y);

// USM APIs

ONEMKL_EXPORT void gemv(oneapi::mkl::device                  libkey,
                        sycl::queue                          &queue,
                        oneapi::mkl::transpose               transpose_val,
                        const fp                             alpha,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        const fp                             *x,
                        const fp                             beta,
                        const fp                             *y,
                        const std::vector<sycl::event>       &dependencies = {});

} //namespace detail
} //namespace spblas
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_SPBLAS_LOADER_HPP_
