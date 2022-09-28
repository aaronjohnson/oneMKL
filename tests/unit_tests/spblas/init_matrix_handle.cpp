/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
//#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

namespace {

int test_declaration_of_matrix_handle_t() {
  oneapi::mkl::sparse::matrix_handle_t *A;
  return 1; // if it compiles and does not crash then PASS
}

int test_init_matrix_handle() {
  oneapi::mkl::sparse::matrix_handle_t *A;

  //  oneapi::mkl::sparse::init_matrix_handle(A);
  oneapi::mkl::sparse::init_matrix_handle(A);

  return 1;
}

class InitMatrixHandleTests : public ::testing::Test {};

  TEST(InitMatrixHandleTestSuite, declarationOfMatrixHandle) {
    EXPECT_TRUEORSKIP(test_declaration_of_matrix_handle_t());
  }

  TEST(InitMatrixHandleTestSuite, initializeMatrixHandle) {
    EXPECT_TRUEORSKIP(test_init_matrix_handle());
  }



} // anonymous namespace
