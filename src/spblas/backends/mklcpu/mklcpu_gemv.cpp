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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/spblas/detail/mklcpu/onemkl_spblas_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace spblas {
namespace mklcpu {

  // Buffer APIs

  void gemv(sycl::queue &queue,
            transpose transpose_flag,
            const float alpha,
            matrix_handle_t handle,
            cl::sycl::buffer<float, 1> &x,
            const float beta,
            cl::sycl::buffer<float, 1> &y) {
    oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y);
  }

  void gemv(sycl::queue &queue,
            transpose transpose_flag,
            const double alpha,
            matrix_handle_t handle,
            cl::sycl::buffer<double, 1> &x,
            const double beta,
            cl::sycl::buffer<double, 1> &y) {
    oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y);
  }

  void gemv(sycl::queue &queue,
            transpose transpose_flag,
            const std::complex<float> alpha,
            matrix_handle_t handle,
            cl::sycl::buffer<std::complex<float>, 1> &x,
            const std::complex<float> beta,
            cl::sycl::buffer<std::complex<float>, 1> &y) {
    oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y);
  }

  void gemv(sycl::queue &queue,
            transpose transpose_flag,
            const std::complex<double> alpha,
            matrix_handle_t handle,
            cl::sycl::buffer<std::complex<double>, 1> &x,
            const std::complex<double> beta,
            cl::sycl::buffer<std::complex<double>, 1> &y) {
    oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y);
  }

  // USM APIs

  sycl::event gemv(cl::sycl::queue &queue,
                   transpose transpose_flag,
                   const float alpha,
                   matrix_handle_t handle,
                   const float *x,
                   const float beta,
                   float *y,
                   const std::vector<cl::sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y, dependencies);
  }

  sycl::event gemv(cl::sycl::queue &queue,
                   transpose transpose_flag,
                   const double alpha,
                   matrix_handle_t handle,
                   const double *x,
                   const double beta,
                   double *y,
                   const std::vector<cl::sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y, dependencies);
  }

  sycl::event gemv(cl::sycl::queue &queue,
                   transpose transpose_flag,
                   const std::complex<float> alpha,
                   matrix_handle_t handle,
                   const std::complex<float> *x,
                   const std::complex<float> beta,
                   std::complex<float> *y,
                   const std::vector<cl::sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y, dependencies);
  }

  sycl::event gemv(cl::sycl::queue &queue,
                   transpose transpose_flag,
                   const std::complex<double> alpha,
                   matrix_handle_t handle,
                   const std::complex<double> *x,
                   const std::complex<double> beta,
                   std::complex<double> *y,
                   const std::vector<cl::sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::gemv(queue, transpose_flag, alpha, handle, x, beta, y, dependencies);
  }

} // namespace mklcpu
} // namespace spblas
} // namespace mkl
} // namespace oneapi
