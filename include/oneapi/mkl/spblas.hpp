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

#ifndef _ONEMKL_SPBLAS_HPP_
#define _ONEMKL_SPBLAS_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
//#include <complex>
//#include <cstdint>

#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/detail/get_device_id.hpp"

#include "oneapi/mkl/spblas/predicates.hpp"

#include "oneapi/mkl/spblas/detail/spblas_loader.hpp"

namespace oneapi {
namespace mkl {
namespace spblas {

  // Buffer APIs

  static inline void gemv(sycl::queue                          &queue,
                          oneapi::mkl::transpose               transpose_val,
                          const fp                             alpha,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          sycl::buffer<fp, 1>                  &x,
                          const fp                             beta,
                          sycl::buffer<fp, 1>                  &y) {
    gemv_precondition(queue, transpose_val, alpha, A_handle, x, beta, y);
    detail::gemv(get_device_id(queue), queue, transpose_val, alpha, A_handle, x, beta, y);
    gemv_postcondition(queue, transpose_val, alpha, A_handle, x, beta, y);
  }
  
  // USM APIs
  
  static inline sycl::event gemv(sycl::queue &queue, std::int64_t n,
                                 const std::complex<float> *x, std::int64_t incx, float *result,
                                 const std::vector<sycl::event> &dependencies = {}) {
    gemv_precondition(queue, transpose_val, alpha, A_handle, x, beta, y, dependencies);
    auto done = detail::gemv(get_device_id(queue), queue, transpose_val, alpha, A_handle, x, beta, y, dependencies);
    gemv_postcondition(queue, transpose_val, alpha, A_handle, x, beta, y, dependencies);
    return done;
  }
  
} //namespace spblas
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_SPBLAS_LOADER_HPP_
