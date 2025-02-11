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

#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/spblas.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"
#include "oneapi/mkl/spblas/detail/mklcpu/onemkl_spblas_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

#define BACKEND mklcpu
#include "oneapi/mkl/spblas/detail/spblas_ct.hxx"
#undef BACKEND

} //namespace sparse
} //namespace mkl
} //namespace oneapi
