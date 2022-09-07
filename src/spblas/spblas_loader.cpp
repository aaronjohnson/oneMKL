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
    namespace sparse {
      namespace detail {

        static oneapi::mkl::detail::table_initializer<domain::spblas, spblas_function_table_t> function_tables;

        void init_matrix_handle(oneapi::mkl::device device,
                                oneapi::mkl::sparse::matrix_handle_t *A) {
          function_tables[device].init_matrix_handle(A);
        }

        void release_matrix_handle(oneapi::mkl::device device,
                                   oneapi::mkl::sparse::matrix_handle_t  A,
                                   const std::vector<sycl::event> &dependencies) {
          function_tables[device].release_matrix_handle(A, dependencies);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t A,
                          std::int32_t                         num_rows,
                          std::int32_t                         num_cols,
                          oneapi::mkl::index_base              index,
                          sycl::buffer<std::int32_t, 1>        &row_ptr,
                          sycl::buffer<std::int32_t, 1>        &col_ind,
                          sycl::buffer<float, 1>               &val) {
          function_tables[device].sset_csr_data_i4_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t A,
                          std::int32_t                         num_rows,
                          std::int32_t                         num_cols,
                          oneapi::mkl::index_base              index,
                          sycl::buffer<std::int32_t, 1>        &row_ptr,
                          sycl::buffer<std::int32_t, 1>        &col_ind,
                          sycl::buffer<double, 1>              &val) {
          function_tables[device].dset_csr_data_i4_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          std::int32_t                          num_rows,
                          std::int32_t                          num_cols,
                          oneapi::mkl::index_base               index,
                          sycl::buffer<std::int32_t, 1>         &row_ptr,
                          sycl::buffer<std::int32_t, 1>         &col_ind,
                          sycl::buffer<std::complex<float>, 1>  &val) {
          function_tables[device].cset_csr_data_i4_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          std::int32_t                          num_rows,
                          std::int32_t                          num_cols,
                          oneapi::mkl::index_base               index,
                          sycl::buffer<std::int32_t, 1>         &row_ptr,
                          sycl::buffer<std::int32_t, 1>         &col_ind,
                          sycl::buffer<std::complex<double>, 1> &val) {
          function_tables[device].zset_csr_data_i4_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          std::int64_t                          num_rows,
                          std::int64_t                          num_cols,
                          oneapi::mkl::index_base               index,
                          sycl::buffer<std::int64_t, 1>         &row_ptr,
                          sycl::buffer<std::int64_t, 1>         &col_ind,
                          sycl::buffer<float, 1>                &val) {
          function_tables[device].sset_csr_data_i8_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          std::int64_t                          num_rows,
                          std::int64_t                          num_cols,
                          oneapi::mkl::index_base               index,
                          sycl::buffer<std::int64_t, 1>         &row_ptr,
                          sycl::buffer<std::int64_t, 1>         &col_ind,
                          sycl::buffer<double, 1>               &val) {
          function_tables[device].dset_csr_data_i8_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          std::int64_t                          num_rows,
                          std::int64_t                          num_cols,
                          oneapi::mkl::index_base               index,
                          sycl::buffer<std::int64_t, 1>         &row_ptr,
                          sycl::buffer<std::int64_t, 1>         &col_ind,
                          sycl::buffer<std::complex<float>, 1>  &val) {
          function_tables[device].cset_csr_data_i8_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          std::int64_t                          num_rows,
                          std::int64_t                          num_cols,
                          oneapi::mkl::index_base               index,
                          sycl::buffer<std::int64_t, 1>         &row_ptr,
                          sycl::buffer<std::int64_t, 1>         &col_ind,
                          sycl::buffer<std::complex<double>, 1> &val) {
          function_tables[device].zset_csr_data_i8_buf(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int32_t                    num_rows,
                          const std::int32_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int32_t                          *row_ptr,
                          std::int32_t                          *col_ind,
                          float                                 *val) {
          function_tables[device].sset_csr_data_i4_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int64_t                    num_rows,
                          const std::int64_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int64_t                          *row_ptr,
                          std::int64_t                          *col_ind,
                          float                                 *val) {
          function_tables[device].sset_csr_data_i8_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int32_t                    num_rows,
                          const std::int32_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int32_t                          *row_ptr,
                          std::int32_t                          *col_ind,
                          double                                *val) {
          function_tables[device].dset_csr_data_i4_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int64_t                    num_rows,
                          const std::int64_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int64_t                          *row_ptr,
                          std::int64_t                          *col_ind,
                          double                                *val) {
          function_tables[device].dset_csr_data_i8_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int32_t                    num_rows,
                          const std::int32_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int32_t                          *row_ptr,
                          std::int32_t                          *col_ind,
                          std::complex<float>                   *val) {
          function_tables[device].cset_csr_data_i4_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int64_t                    num_rows,
                          const std::int64_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int64_t                          *row_ptr,
                          std::int64_t                          *col_ind,
                          std::complex<float>                   *val) {
          function_tables[device].cset_csr_data_i8_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int32_t                    num_rows,
                          const std::int32_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int32_t                          *row_ptr,
                          std::int32_t                          *col_ind,
                          std::complex<double>                  *val) {
          function_tables[device].zset_csr_data_i4_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        void set_csr_data(oneapi::mkl::device,
                          oneapi::mkl::sparse::matrix_handle_t  A,
                          const std::int64_t                    num_rows,
                          const std::int64_t                    num_cols,
                          oneapi::mkl::index_base               index,
                          std::int64_t                          *row_ptr,
                          std::int64_t                          *col_ind,
                          std::complex<double>                  *val) {
          function_tables[device].zset_csr_data_i8_usm(A, num_rows, num_cols, index, row_ptr, col_ind, val);
        }

        // Buffer APIs

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const float                          alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  sycl::buffer<float, 1>               &x,
                  const float                          beta,
                  sycl::buffer<float, 1>               &y) {
          function_tables[device].sgemv_buf(queue, transpose, alpha, A, x, beta, y);
        }

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const double                         alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  sycl::buffer<double, 1>              &x,
                  const double                         beta,
                  sycl::buffer<double, 1>              &y) {
          function_tables[device].dgemv_buf(queue, transpose, alpha, A, x, beta, y);
        }

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const std::complex<float>            alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  sycl::buffer<std::complex<float>, 1> &x,
                  const std::complex<float>            beta,
                  sycl::buffer<std::complex<float>, 1> &y) {
          function_tables[device].cgemv_buf(queue, transpose, alpha, A, x, beta, y);
        }

        void gemv(oneapi::mkl::device                   device,
                  sycl::queue                           &queue,
                  oneapi::mkl::transpose                transpose,
                  const std::complex<double>            alpha,
                  oneapi::mkl::sparse::matrix_handle_t  A,
                  sycl::buffer<std::complex<double>, 1> &x,
                  const std::complex<double>            beta,
                  sycl::buffer<std::complex<double>, 1> &y) {
          function_tables[device].zgemv_buf(queue, transpose, alpha, A, x, beta, y);
        }

        // USM APIs

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const float                          alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  const float                          *x,
                  const float                          beta,
                  const float                          *y,
                  const std::vector<sycl::event>       &dependencies) {
          function_tables[device].sgemv_usm(queue, transpose, alpha, A, x, beta, y, dependencies);
        }

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const double                         alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  const double                         *x,
                  const double                         beta,
                  const double                         *y,
                  const std::vector<sycl::event>       &dependencies) {
          function_tables[device].dgemv_usm(queue, transpose, alpha, A, x, beta, y, dependencies);
        }

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const std::complex<float>            alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  const std::complex<float>            *x,
                  const std::complex<float>            beta,
                  const std::complex<float>            *y,
                  const std::vector<sycl::event>       &dependencies) {
          function_tables[device].cgemv_usm(queue, transpose, alpha, A, x, beta, y, dependencies);
        }

        void gemv(oneapi::mkl::device                  device,
                  sycl::queue                          &queue,
                  oneapi::mkl::transpose               transpose,
                  const std::complex<double>           alpha,
                  oneapi::mkl::sparse::matrix_handle_t A,
                  const std::complex<double>           *x,
                  const std::complex<double>           beta,
                  const std::complex<double>           *y,
                  const std::vector<sycl::event>       &dependencies)  {
          function_tables[device].zgemv_usm(queue, transpose, alpha, A, x, beta, y, dependencies);
        }
      } //namespace detail
    } //namespace sparse
  } //namespace mkl
} //namespace oneapi
