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

// Buffer APIs

static inline void gemv(backend_selector<backend::BACKEND>    selector,
                        sycl::queue                           &queue,
                        oneapi::mkl::transpose                transpose,
                        const float                           alpha,
                        oneapi::mkl::sparse::matrix_handle_t  A,
                        sycl::buffer<float, 1>                &x,
                        const float                           beta,
                        sycl::buffer<float, 1>                &y) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y);
}

static inline void gemv(backend_selector<backend::BACKEND>    selector,
                        oneapi::mkl::transpose                transpose,
                        const double                          alpha,
                        oneapi::mkl::sparse::matrix_handle_t  A,
                        sycl::buffer<double, 1>               &x,
                        const double                          beta,
                        sycl::buffer<double, 1>               &y) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y);
}

static inline void gemv(backend_selector<backend::BACKEND>    selector,
                        oneapi::mkl::transpose                transpose,
                        const std::complex<float>             alpha,
                        oneapi::mkl::sparse::matrix_handle_t  A,
                        sycl::buffer<std::complex<float>, 1>  &x,
                        const std::complex<float>             beta,
                        sycl::buffer<std::complex<float>, 1>  &y) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y);
}

static inline void gemv(backend_selector<backend::BACKEND>    selector,
                        oneapi::mkl::transpose                transpose,
                        const std::complex<double>            alpha,
                        oneapi::mkl::sparse::matrix_handle_t  A,
                        sycl::buffer<std::complex<double>, 1> &x,
                        const std::complex<double>            beta,
                        sycl::buffer<std::complex<double>, 1> &y) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y);
}

// USM APIs

static inline void gemv(backend_selector<backend::BACKEND>   selector,
                        oneapi::mkl::transpose               transpose,
                        const float                          alpha,
                        oneapi::mkl::sparse::matrix_handle_t A,
                        const float                          *x,
                        const float                          beta,
                        const float                          *y,
                        const std::vector<sycl::event>       &dependencies) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y, dependencies);
}

static inline void gemv(backend_selector<backend::BACKEND>   selector,
                        oneapi::mkl::transpose               transpose,
                        const double                         alpha,
                        oneapi::mkl::sparse::matrix_handle_t A,
                        const double                         *x,
                        const double                         beta,
                        const double                         *y,
                        const std::vector<sycl::event>       &dependencies) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y, dependencies);
}

static inline void gemv(backend_selector<backend::BACKEND>   selector,
                        oneapi::mkl::transpose               transpose,
                        const std::complex<float>            alpha,
                        oneapi::mkl::sparse::matrix_handle_t A,
                        const std::complex<float>            *x,
                        const std::complex<float>            beta,
                        const std::complex<float>            *y,
                        const std::vector<sycl::event>       &dependencies) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y, dependencies);
}

static inline void gemv(backend_selector<backend::BACKEND>   selector,
                        oneapi::mkl::transpose               transpose,
                        const std::complex<double>           alpha,
                        oneapi::mkl::sparse::matrix_handle_t A,
                        const std::complex<double>           *x,
                        const std::complex<double>           beta,
                        const std::complex<double>           *y,
                        const std::vector<sycl::event>       &dependencies) {
  oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(),
                                     transpose, alpha, A, x, beta, y, dependencies);
}
