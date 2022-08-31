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

#include "oneapi/mkl/spblas/detail/spblas_loader.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

  void init_matrix_handle (oneapi::mkl::sparse::matrix_handle_t *handle);

  void release_matrix_handle (oneapi::mkl::sparse::matrix_handle_t  handle,
                              const std::vector<sycl::event>        &dependencies = {});

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int32_t                          num_rows,
                     std::int32_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int32_t, 1>         &row_ptr,
                     sycl::buffer<std::int32_t, 1>         &col_ind,
                     sycl::buffer<float, 1>                &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int32_t                          num_rows,
                     std::int32_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int32_t, 1>         &row_ptr,
                     sycl::buffer<std::int32_t, 1>         &col_ind,
                     sycl::buffer<double, 1>               &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int32_t                          num_rows,
                     std::int32_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int32_t, 1>         &row_ptr,
                     sycl::buffer<std::int32_t, 1>         &col_ind,
                     sycl::buffer<std::complex<float>, 1>  &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int32_t                          num_rows,
                     std::int32_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int32_t, 1>         &row_ptr,
                     sycl::buffer<std::int32_t, 1>         &col_ind,
                     sycl::buffer<std::complex<double>, 1> &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int64_t                          num_rows,
                     std::int64_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int64_t, 1>         &row_ptr,
                     sycl::buffer<std::int64_t, 1>         &col_ind,
                     sycl::buffer<float, 1>                &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int64_t                          num_rows,
                     std::int64_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int64_t, 1>         &row_ptr,
                     sycl::buffer<std::int64_t, 1>         &col_ind,
                     sycl::buffer<double, 1>                   &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int64_t                          num_rows,
                     std::int64_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int64_t, 1>         &row_ptr,
                     sycl::buffer<std::int64_t, 1>         &col_ind,
                     sycl::buffer<std::complex<float>, 1>  &val);

  void set_csr_data (oneapi::mkl::sparse::matrix_handle_t  handle,
                     std::int64_t                          num_rows,
                     std::int64_t                          num_cols,
                     oneapi::mkl::index_base               index,
                     sycl::buffer<std::int64_t, 1>         &row_ptr,
                     sycl::buffer<std::int64_t, 1>         &col_ind,
                     sycl::buffer<std::complex<double>, 1> &val);

} //namespace sparse
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_SPBLAS_LOADER_HPP_
