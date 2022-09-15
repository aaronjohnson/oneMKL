
static inline void init_matrix_handle(oneapi::mkl::sparse::matrix_handle_t *A) {
  oneapi::mkl::sparse::BACKEND::init_matrix_handle(A);
}

static inline void release_matrix_handle(oneapi::mkl::sparse::matrix_handle_t *A,
                                         const std::vector<sycl::event>       &dependencies) {
  oneapi::mkl::sparse::BACKEND::release_matrix_handle(A, dependencies);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                sycl::buffer<std::int32_t, 1>        &row_ptr,
                                sycl::buffer<std::int32_t, 1>        &col_ind,
                                sycl::buffer<float, 1>               &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                sycl::buffer<std::int64_t, 1>        &row_ptr,
                                sycl::buffer<std::int64_t, 1>        &col_ind,
                                sycl::buffer<float, 1>               &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                sycl::buffer<std::int32_t, 1>        &row_ptr,
                                sycl::buffer<std::int32_t, 1>        &col_ind,
                                sycl::buffer<double, 1>              &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                sycl::buffer<std::int64_t, 1>        &row_ptr,
                                sycl::buffer<std::int64_t, 1>        &col_ind,
                                sycl::buffer<double, 1>              &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                sycl::buffer<std::int32_t, 1>        &row_ptr,
                                sycl::buffer<std::int32_t, 1>        &col_ind,
                                sycl::buffer<std::complex<float>, 1> &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                sycl::buffer<std::int64_t, 1>        &row_ptr,
                                sycl::buffer<std::int64_t, 1>        &col_ind,
                                sycl::buffer<std::complex<float>, 1> &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t  A,
                                const std::int32_t                    num_rows,
                                const std::int32_t                    num_cols,
                                oneapi::mkl::index_base               index_base,
                                sycl::buffer<std::int32_t, 1>         &row_ptr,
                                sycl::buffer<std::int32_t, 1>         &col_ind,
                                sycl::buffer<std::complex<double>, 1> &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t  A,
                                const std::int64_t                    num_rows,
                                const std::int64_t                    num_cols,
                                oneapi::mkl::index_base               index_base,
                                sycl::buffer<std::int64_t, 1>         &row_ptr,
                                sycl::buffer<std::int64_t, 1>         &col_ind,
                                sycl::buffer<std::complex<double>, 1> &val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int32_t                         *row_ptr,
                                std::int32_t                         *col_ind,
                                float                                *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int64_t                         *row_ptr,
                                std::int64_t                         *col_ind,
                                float                                *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int32_t                         *row_ptr,
                                std::int32_t                         *col_ind,
                                double                               *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int64_t                         *row_ptr,
                                std::int64_t                         *col_ind,
                                double                               *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int32_t                         *row_ptr,
                                std::int32_t                         *col_ind,
                                std::complex<float>                  *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int64_t                         *row_ptr,
                                std::int64_t                         *col_ind,
                                std::complex<float>                  *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int32_t                   num_rows,
                                const std::int32_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int32_t                         *row_ptr,
                                std::int32_t                         *col_ind,
                                std::complex<double>                 *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}

static inline void set_csr_data(oneapi::mkl::sparse::matrix_handle_t A,
                                const std::int64_t                   num_rows,
                                const std::int64_t                   num_cols,
                                oneapi::mkl::index_base              index_base,
                                std::int64_t                         *row_ptr,
                                std::int64_t                         *col_ind,
                                std::complex<double>                 *val) {
  oneapi::mkl::sparse::BACKEND::set_csr_data(A, num_rows, num_cols, index_base, row_ptr, col_ind, val);
}
