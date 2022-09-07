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

  void (*init_matrix_handle)(oneapi::mkl::sparse::matrix_handle_t *A);
  void (*release_matrix_handle)(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::vector<sycl::event> &dependencies);

  // Buffer APIs

  void (*sset_csr_data_i4_buf)(oneapi::mkl::sparse::matrix_handle_t A,
                               std::int32_t                         num_rows,
                               std::int32_t                         num_cols,
                               oneapi::mkl::index_base              index_base,
                               sycl::buffer<std::int32_t, 1>        &row_ptr,
                               sycl::buffer<std::int32_t, 1>        &col_ind,
                               sycl::buffer<float, 1>               &val);

  void (*sset_csr_data_i8_buf)(oneapi::mkl::sparse::matrix_handle_t A,
                               std::int64_t                         num_rows,
                               std::int64_t                         num_cols,
                               oneapi::mkl::index_base              index_base,
                               sycl::buffer<std::int64_t, 1>        &row_ptr,
                               sycl::buffer<std::int64_t, 1>        &col_ind,
                               sycl::buffer<float, 1>               &val);

  void (*dset_csr_data_i4_buf)(oneapi::mkl::sparse::matrix_handle_t A,
                               std::int32_t                         num_rows,
                               std::int32_t                         num_cols,
                               oneapi::mkl::index_base              index_base,
                               sycl::buffer<std::int32_t, 1>        &row_ptr,
                               sycl::buffer<std::int32_t, 1>        &col_ind,
                               sycl::buffer<double, 1>              &val);

  void (*dset_csr_data_i8_buf)(oneapi::mkl::sparse::matrix_handle_t A,
                               std::int64_t                         num_rows,
                               std::int64_t                         num_cols,
                               oneapi::mkl::index_base              index_base,
                               sycl::buffer<std::int64_t, 1>        &row_ptr,
                               sycl::buffer<std::int64_t, 1>        &col_ind,
                               sycl::buffer<double, 1>              &val);

  void (*cset_csr_data_i4_buf)(oneapi::mkl::sparse::matrix_handle_t A,
                               std::int32_t                         num_rows,
                               std::int32_t                         num_cols,
                               oneapi::mkl::index_base              index_base,
                               sycl::buffer<std::int32_t, 1>        &row_ptr,
                               sycl::buffer<std::int32_t, 1>        &col_ind,
                               sycl::buffer<std::complex<float>, 1> &val);

  void (*cset_csr_data_i8_buf)(oneapi::mkl::sparse::matrix_handle_t A,
                               std::int64_t                         num_rows,
                               std::int64_t                         num_cols,
                               oneapi::mkl::index_base              index_base,
                               sycl::buffer<std::int64_t, 1>        &row_ptr,
                               sycl::buffer<std::int64_t, 1>        &col_ind,
                               sycl::buffer<std::complex<float>, 1> &val);


  void (*zset_csr_data_i4_buf)(oneapi::mkl::sparse::matrix_handle_t  A,
                               std::int32_t                          num_rows,
                               std::int32_t                          num_cols,
                               oneapi::mkl::index_base               index_base,
                               sycl::buffer<std::int32_t, 1>         &row_ptr,
                               sycl::buffer<std::int32_t, 1>         &col_ind,
                               sycl::buffer<std::complex<double>, 1> &val);

  void (*zset_csr_data_i8_buf)(oneapi::mkl::sparse::matrix_handle_t  A,
                               std::int64_t                          num_rows,
                               std::int64_t                          num_cols,
                               oneapi::mkl::index_base               index_base,
                               sycl::buffer<std::int64_t, 1>         &row_ptr,
                               sycl::buffer<std::int64_t, 1>         &col_ind,
                               sycl::buffer<std::complex<double>, 1> &val);

  void (*sgemv_buf)(sycl::queue                          &queue,
                    oneapi::mkl::transpose               transpose,
                    const float                          alpha,
                    oneapi::mkl::sparse::matrix_t        A,
                    sycl::buffer<float, 1>               &x,
                    const float                          beta,
                    sycl::buffer<float, 1>               &y);

  void (*dgemv_buf)(sycl::queue                          &queue,
                    oneapi::mkl::transpose               transpose,
                    const double                         alpha,
                    oneapi::mkl::sparse::matrix_handle_t A,
                    sycl::buffer<double, 1>              &x,
                    const double                         beta,
                    sycl::buffer<double, 1>              &y);

  void (*cgemv_buf)(sycl::queue                          &queue,
                    oneapi::mkl::transpose               transpose,
                    const std::complex<float>            alpha,
                    oneapi::mkl::sparse::matrix_handle_t A,
                    sycl::buffer<std::complex<float>, 1> &x,
                    const std::complex<float>            beta,
                    sycl::buffer<std::complex<float>, 1> &y);

  void (*zgemv_buf)(sycl::queue                           &queue,
                    oneapi::mkl::transpose                transpose,
                    const std::complex<double>            alpha,
                    oneapi::mkl::sparse::matrix_handle_t  A,
                    sycl::buffer<std::complex<double>, 1> &x,
                    const std::complex<double>            beta,
                    sycl::buffer<std::complex<double>, 1> &y);

  // USM APIs

  void (*sset_csr_data_i4_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int32_t                   num_rows,
                               const std::int32_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int32_t                         *row_ptr,
                               std::int32_t                         *col_ind,
                               float                                *val);

  void (*sset_csr_data_i8_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int64_t                   num_rows,
                               const std::int64_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int64_t                         *row_ptr,
                               std::int64_t                         *col_ind,
                               float                                *val);

  void (*dset_csr_data_i4_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int32_t                   num_rows,
                               const std::int32_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int32_t                         *row_ptr,
                               std::int32_t                         *col_ind,
                               double                               *val);

  void (*dset_csr_data_i8_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int64_t                   num_rows,
                               const std::int64_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int64_t                         *row_ptr,
                               std::int64_t                         *col_ind,
                               double                               *val);

  void (*cset_csr_data_i4_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int32_t                   num_rows,
                               const std::int32_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int32_t                         *row_ptr,
                               std::int32_t                         *col_ind,
                               std::complex<float>                  *val);

  void (*cset_csr_data_i8_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int64_t                   num_rows,
                               const std::int64_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int64_t                         *row_ptr,
                               std::int64_t                         *col_ind,
                               std::complex<float>                  *val);

  void (*zset_csr_data_i4_usm)(oneapi::mkl::sparse::matrix_handle_t A,
                               const std::int32_t                   num_rows,
                               const std::int32_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int32_t                         *row_ptr,
                               std::int32_t                         *col_ind,
                               std::complex<double>                 *val);

  void (*zset_csr_data_i8_usm)(oneapi::mkl::sparse::matrix_handle_t  A,
                               const std::int64_t                   num_rows,
                               const std::int64_t                   num_cols,
                               oneapi::mkl::index_base              index_base,
                               std::int64_t                         *row_ptr,
                               std::int64_t                         *col_ind,
                               std::complex<double>                 *val);

  sycl::event (*sgemv_usm)(sycl::queue                          &queue,
                           oneapi::mkl::transpose               transpose,
                           const float                          alpha,
                           oneapi::mkl::sparse::matrix_handle_t A,
                           const float                          *x,
                           const float                          beta,
                           const float                          *y,
                           const std::vector<sycl::event>       &dependencies);

  sycl::event (*dgemv_usm)(sycl::queue                          &queue,
                           oneapi::mkl::transpose               transpose,
                           const double                         alpha,
                           oneapi::mkl::sparse::matrix_handle_t A,
                           const double                         *x,
                           const double                         beta,
                           const double                         *y,
                           const std::vector<sycl::event>       &dependencies);

  sycl::event (*cgemv_usm)(sycl::queue                          &queue,
                           oneapi::mkl::transpose               transpose,
                           const std::complex<float>            alpha,
                           oneapi::mkl::sparse::matrix_handle_t A,
                           const std::complex<float>            *x,
                           const std::complex<float>            beta,
                           const std::complex<float>            *y,
                           const std::vector<sycl::event>       &dependencies);

  sycl::event (*zgemv_usm)(sycl::queue                          &queue,
                           oneapi::mkl::transpose               transpose,
                           const std::complex<double>           alpha,
                           oneapi::mkl::sparse::matrix_handle_t A,
                           const std::complex<double>           *x,
                           const std::complex<double>           beta,
                           const std::complex<double>           *y,
                           const std::vector<sycl::event>       &dependencies);
} spblas_function_table_t;

#endif //_SPBLAS_FUNCTION_TABLE_HPP_
