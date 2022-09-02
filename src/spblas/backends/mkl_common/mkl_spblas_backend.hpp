/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

struct matrix_handle;
typedef struct matrix_handle *matrix_handle_t;

 void init_matrix_handle(oneapi::mkl::sparse::matrix_handle_t *handle);

 void release_matrix_handle(oneapi::mkl::sparse::matrix_handle_t *handle,
                      const std::vector<cl::sycl::event> &dependencies);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<float, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<float, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<double, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<double, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<std::complex<float>, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<std::complex<float>, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<std::complex<double>, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<std::complex<double>, 1> &val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             float *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             float *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             double *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             double *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             std::complex<float> *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             std::complex<float> *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             std::complex<double> *val);

 void set_csr_data(oneapi::mkl::sparse::matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             oneapi::mkl::index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             std::complex<double> *val);


 void gemv(sycl::queue &queue,
                     oneapi::mkl::transpose transpose_flag,
                     const float alpha,
                     oneapi::mkl::sparse::matrix_handle_t handle,
                     sycl::buffer<float, 1> &x,
                     const float beta,
                     sycl::buffer<float, 1> &y);

 void gemv(sycl::queue &queue,
                     oneapi::mkl::transpose transpose_flag,
                     const double alpha,
                     oneapi::mkl::sparse::matrix_handle_t handle,
                     sycl::buffer<double, 1> &x,
                     const double beta,
                     sycl::buffer<double, 1> &y);

 void gemv(sycl::queue &queue,
                     oneapi::mkl::transpose transpose_flag,
                     const std::complex<float> alpha,
                     oneapi::mkl::sparse::matrix_handle_t handle,
                     sycl::buffer<std::complex<float>, 1> &x,
                     const std::complex<float> beta,
                     sycl::buffer<std::complex<float>, 1> &y);

 void gemv(sycl::queue &queue,
                     oneapi::mkl::transpose transpose_flag,
                     const std::complex<double> alpha,
                     oneapi::mkl::sparse::matrix_handle_t handle,
                     sycl::buffer<std::complex<double>, 1> &x,
                     const std::complex<double> beta,
                     sycl::buffer<std::complex<double>, 1> &y);

 sycl::event gemv(sycl::queue &queue,
                                oneapi::mkl::transpose transpose_flag,
                                const float alpha,
                                oneapi::mkl::sparse::matrix_handle_t handle,
                                const float *x,
                                const float beta,
                                float *y,
                                const std::vector<sycl::event> &dependencies);

 sycl::event gemv(sycl::queue &queue,
                                oneapi::mkl::transpose transpose_flag,
                                const double alpha,
                                oneapi::mkl::sparse::matrix_handle_t handle,
                                const double *x,
                                const double beta,
                                double *y,
                                const std::vector<sycl::event> &dependencies);

 sycl::event gemv(sycl::queue &queue,
                                oneapi::mkl::transpose transpose_flag,
                                const std::complex<float> alpha,
                                oneapi::mkl::sparse::matrix_handle_t handle,
                                const std::complex<float> *x,
                                const std::complex<float> beta,
                                std::complex<float> *y,
                                const std::vector<sycl::event> &dependencies);

 sycl::event gemv(sycl::queue &queue,
                                oneapi::mkl::transpose transpose_flag,
                                const std::complex<double> alpha,
                                oneapi::mkl::sparse::matrix_handle_t handle,
                                const std::complex<double> *x,
                                const std::complex<double> beta,
                                std::complex<double> *y,
                                const std::vector<sycl::event> &dependencies);

}
}
}
