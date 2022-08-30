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

#ifndef _SPBLAS_FUNCTION_TABLE_HPP_
#define _SPBLAS_FUNCTION_TABLE_HPP_

#include <complex>
#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl/types.hpp"

typedef struct {
    int version;

    // Buffer APIs

    void (*sgemv_sycl)(sycl::queue                          &queue,
                       oneapi::mkl::transpose               transpose_val,
                       const float                          alpha,
                       oneapi::mkl::sparse::matrix_handle_t A_handle,
                       sycl::buffer<float, 1>               &x,
                       const float                          beta,
                       sycl::buffer<float, 1>               &y
                      );

    void (*dgemv_sycl)(sycl::queue                          &queue,
                       oneapi::mkl::transpose               transpose_val,
                       const double                         alpha,
                       oneapi::mkl::sparse::matrix_handle_t A_handle,
                       sycl::buffer<double, 1>              &x,
                       const double                         beta,
                       sycl::buffer<double, 1>              &y
                      );

    void (*cgemv_sycl)(sycl::queue                          &queue,
                       oneapi::mkl::transpose               transpose_val,
                       const std::complex<float>            alpha,
                       oneapi::mkl::sparse::matrix_handle_t A_handle,
                       sycl::buffer<std::complex<float>, 1> &x,
                       const std::complex<float>            beta,
                       sycl::buffer<std::complex<float>, 1> &y
                      );

    void (*zgemv_sycl)(sycl::queue                           &queue,
                       oneapi::mkl::transpose                transpose_val,
                       const std::complex<double>            alpha,
                       oneapi::mkl::sparse::matrix_handle_t  A_handle,
                       sycl::buffer<std::complex<double>, 1> &x,
                       const std::complex<double>            beta,
                       sycl::buffer<std::complex<double>, 1> &y
                      );

    // USM APIs

    sycl::event (*sgemv_usm_sycl)(sycl::queue                          &queue,
                                  oneapi::mkl::transpose               transpose_val,
                                  const float                          alpha,
                                  oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  const float                          *x,
                                  const float                          beta,
                                  const float                          *y,
                                  const std::vector<sycl::event>       &dependencies = {}
                                 );

    sycl::event (*dgemv_usm_sycl)(sycl::queue                          &queue,
                                  oneapi::mkl::transpose               transpose_val,
                                  const double                         alpha,
                                  oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  const double                         *x,
                                  const double                         beta,
                                  const double                         *y,
                                  const std::vector<sycl::event>       &dependencies = {}
                                 );

    sycl::event (*cgemv_usm_sycl)(sycl::queue                          &queue,
                                  oneapi::mkl::transpose               transpose_val,
                                  const std::complex<float>            alpha,
                                  oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  const std::complex<float>            *x,
                                  const std::complex<float>            beta,
                                  const std::complex<float>            *y,
                                  const std::vector<sycl::event>       &dependencies = {}
                                 );

    sycl::event (*zgemv_usm_sycl)(sycl::queue                          &queue,
                                  oneapi::mkl::transpose               transpose_val,
                                  const std::complex<double>           alpha,
                                  oneapi::mkl::sparse::matrix_handle_t A_handle,
                                  const std::complex<double>           *x,
                                  const std::complex<double>           beta,
                                  const std::complex<double>           *y,
                                  const std::vector<sycl::event>       &dependencies = {}
                                 );

} spblas_function_table_t;

#endif //_SPBLAS_FUNCTION_TABLE_HPP_
