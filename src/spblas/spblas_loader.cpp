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

#include "oneapi/mkl/spblas/detail/spblas_loader.hpp"

#include "function_table_initializer.hpp"
#include "spblas/function_table.hpp"

namespace oneapi {
namespace mkl {
namespace spblas {
namespace detail {

static oneapi::mkl::detail::table_initializer<domain::spblas, spblas_function_table_t> function_tables;

// Buffer APIs

  void gemv(oneapi::mkl::device                  libkey,
            sycl::queue                          &queue,
            oneapi::mkl::transpose               transpose_val,
            const float                          alpha,
            oneapi::mkl::sparse::matrix_handle_t A_handle,
            sycl::buffer<float, 1>               &x,
            const float                          beta,
            sycl::buffer<float, 1>               &y) {
    function_tables[libkey].sgemv_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

  void gemv(oneapi::mkl::device                  libkey,
            sycl::queue                          &queue,
            oneapi::mkl::transpose               transpose_val,
            const double                         alpha,
            oneapi::mkl::sparse::matrix_handle_t A_handle,
            sycl::buffer<double, 1>              &x,
            const double                         beta,
            sycl::buffer<double, 1>              &y) {
    function_tables[libkey].dgemv_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

  void gemv(oneapi::mkl::device                   libkey,
            sycl::queue                           &queue,
            oneapi::mkl::transpose                transpose_val,
            const std::complex<float>             alpha,
            oneapi::mkl::sparse::matrix_handle_t  A_handle,
            sycl::buffer<std::complex<float>, 1>  &x,
            const std::complex<float>             beta,
            sycl::buffer<std::complex<float>, 1>  &y) {
    function_tables[libkey].cgemv_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

  void gemv(oneapi::mkl::device                   libkey,
            sycl::queue                           &queue,
            oneapi::mkl::transpose                transpose_val,
            const std::complex<double>            alpha,
            oneapi::mkl::sparse::matrix_handle_t  A_handle,
            sycl::buffer<std::complex<double>, 1> &x,
            const std::complex<double>            beta,
            sycl::buffer<std::complex<double>, 1> &y) {
    function_tables[libkey].zgemv_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

// USM APIs

  void gemv(oneapi::mkl::device                  libkey,
            sycl::queue                          &queue,
            oneapi::mkl::transpose               transpose_val,
            const float                          alpha,
            oneapi::mkl::sparse::matrix_handle_t A_handle,
            const float                          *x,
            const float                          beta,
            const float                          *y,
            const std::vector<sycl::event>       &dependencies = {}) {
    function_tables[libkey].sgemv_usm_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

  void gemv(oneapi::mkl::device                  libkey,
            sycl::queue                          &queue,
            oneapi::mkl::transpose               transpose_val,
            const double                         alpha,
            oneapi::mkl::sparse::matrix_handle_t A_handle,
            const double                         *x,
            const double                         beta,
            const double                         *y,
            const std::vector<sycl::event>       &dependencies = {}) {
    function_tables[libkey].dgemv_usm_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

  void gemv(oneapi::mkl::device                  libkey,
            sycl::queue                          &queue,
            oneapi::mkl::transpose               transpose_val,
            const std::complex<float>            alpha,
            oneapi::mkl::sparse::matrix_handle_t A_handle,
            const std::complex<float>            *x,
            const std::complex<float>            beta,
            const std::complex<float>            *y,
            const std::vector<sycl::event>       &dependencies = {}) {
    function_tables[libkey].cgemv_usm_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

  void gemv(oneapi::mkl::device                  libkey,
            sycl::queue                          &queue,
            oneapi::mkl::transpose               transpose_val,
            const std::complex<double>           alpha,
            oneapi::mkl::sparse::matrix_handle_t A_handle,
            const std::complex<double>           *x,
            const std::complex<double>           beta,
            const std::complex<double>           *y,
            const std::vector<sycl::event>       &dependencies = {})  {
    function_tables[libkey].zgemv_usm_sycl(queue, transpose_val, alpha, A_handle, x, beta, y);
  }

} //namespace detail
} //namespace row_major
} //namespace blas
} //namespace mkl
} //namespace oneapi
