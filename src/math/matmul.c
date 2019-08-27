#include "src/math/matmul.h"
#include "src/basic/broadcast.h"
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

int64_t max_int64_(int64_t a, int64_t b) { return a > b ? a : b; }

// get the strides of dims
int64_t batch_strides(int64_t *dims, int64_t ndim, int64_t *strides,
                      int64_t nstride) {
  int64_t stride = 1;
  for (int64_t i = 0; i < nstride; ++i) {
    // ndim - 1 - i >= 0 to make sure it is not out bound.
    if (ndim - 1 - i >= 0 && dims[ndim - 1 - i] != 1) {
      strides[nstride - 1 - i] = stride;
      stride *= dims[ndim - 1 - i];
    } else {
      strides[nstride - 1 - i] = 0;
    }
  }
}

// kernel of vector-vector multiply
#define VV_KERNEL(typename, X, Y, Z, L)                            \
  for (int i = 0; i < L; ++i) {                                    \
    ((typename *)Z)[0] += ((typename *)X)[i] * ((typename *)Y)[i]; \
  }

// kernel of matrix-vector multiply
#define MV_KERNEL(typename, A, X, B, M, N)                                   \
  for (int i = 0; i < M; ++i) {                                              \
    for (int j = 0; j < N; ++j) {                                            \
      ((typename *)B)[i] += ((typename *)A)[i * N + j] * ((typename *)X)[j]; \
    }                                                                        \
  }

// kernel of matrix-matrix multiply
#define MM_KERNEL(typename, A, B, C, M, K, N)                        \
  for (int i = 0; i < M; ++i) {                                      \
    for (int j = 0; j < N; ++j) {                                    \
      for (int q = 0; q < K; ++q) {                                  \
        ((typename *)C)[i * N + j] +=                                \
            ((typename *)A)[i * K + q] * ((typename *)B)[q * N + j]; \
      }                                                              \
    }                                                                \
  }

// kernel of batch matrix-matrix multiply
#define BATCH_MM_KERNEL(typename, As, Bs, Cs, M, K, N, n_batch) \
  for (int64_t bth = 0; bth < n_batch; ++bth) {                 \
    typename *A = *((typename **)As + bth);                     \
    typename *B = *((typename **)Bs + bth);                     \
    typename *C = *((typename **)Cs + bth);                     \
    MM_KERNEL(typename, A, B, C, M, K, N);                      \
  }

// choose vv kernel according to dtype.code
void vv_template(DataType dtype, void *X, void *Y, void *Z, int64_t L,
                 Status *status) {
  *status = STATUS_SUCCESS;
  switch (dtype.code) {
    case TYPE_INT8: {
      VV_KERNEL(int8_t, X, Y, Z, L);
      break;
    }
    case TYPE_UINT8: {
      VV_KERNEL(uint8_t, X, Y, Z, L);
      break;
    }
    case TYPE_INT16: {
      VV_KERNEL(int16_t, X, Y, Z, L);
      break;
    }
    case TYPE_UINT16: {
      VV_KERNEL(uint16_t, X, Y, Z, L);
      break;
    }
    case TYPE_INT32: {
      VV_KERNEL(int32_t, X, Y, Z, L);
      break;
    }
    case TYPE_UINT32: {
      VV_KERNEL(uint32_t, X, Y, Z, L);
      break;
    }
    case TYPE_INT64: {
      VV_KERNEL(int64_t, X, Y, Z, L);
      break;
    }
    case TYPE_UINT64: {
      VV_KERNEL(uint64_t, X, Y, Z, L);
      break;
    }
    case TYPE_FLOAT: {
      VV_KERNEL(float, X, Y, Z, L);
      break;
    }
    case TYPE_DOUBLE: {
      VV_KERNEL(double, X, Y, Z, L);
      break;
    }
    default:
      *status = STATUS_NOT_SUPPORTED;
  }
}

// choose mv kernel according to dtype.code
void mv_template(DataType dtype, void *A, void *X, void *B, int64_t M,
                 int64_t N, Status *status) {
  *status = STATUS_SUCCESS;
  switch (dtype.code) {
    case TYPE_INT8: {
      MV_KERNEL(int8_t, A, X, B, M, N);
      break;
    }
    case TYPE_UINT8: {
      MV_KERNEL(uint8_t, A, X, B, M, N);
      break;
    }
    case TYPE_INT16: {
      MV_KERNEL(int16_t, A, X, B, M, N);
      break;
    }
    case TYPE_UINT16: {
      MV_KERNEL(uint16_t, A, X, B, M, N);
      break;
    }
    case TYPE_INT32: {
      MV_KERNEL(int32_t, A, X, B, M, N);
      break;
    }
    case TYPE_UINT32: {
      MV_KERNEL(uint32_t, A, X, B, M, N);
      break;
    }
    case TYPE_INT64: {
      MV_KERNEL(int64_t, A, X, B, M, N);
      break;
    }
    case TYPE_UINT64: {
      MV_KERNEL(uint64_t, A, X, B, M, N);
      break;
    }
    case TYPE_FLOAT: {
      MV_KERNEL(float, A, X, B, M, N);
      break;
    }
    case TYPE_DOUBLE: {
      MV_KERNEL(double, A, X, B, M, N);
      break;
    }
    default:
      *status = STATUS_NOT_SUPPORTED;
  }
}

// choose mm kernel according to dtype.code
void mm_template(DataType dtype, void *A, void *B, void *C, int64_t M,
                 int64_t K, int64_t N, Status *status) {
  *status = STATUS_SUCCESS;
  switch (dtype.code) {
    case TYPE_INT8: {
      MM_KERNEL(int8_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_UINT8: {
      MM_KERNEL(uint8_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_INT16: {
      MM_KERNEL(int16_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_UINT16: {
      MM_KERNEL(uint16_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_INT32: {
      MM_KERNEL(int32_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_UINT32: {
      MM_KERNEL(uint32_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_INT64: {
      MM_KERNEL(int64_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_UINT64: {
      MM_KERNEL(uint64_t, A, B, C, M, K, N);
      break;
    }
    case TYPE_FLOAT: {
      MM_KERNEL(float, A, B, C, M, K, N);
      break;
    }
    case TYPE_DOUBLE: {
      MM_KERNEL(double, A, B, C, M, K, N);
      break;
    }
    default:
      *status = STATUS_NOT_SUPPORTED;
  }
}

// choose batch_mm kernel according to dtype.code
void batch_mm_template(DataType dtype, void **As, void **Bs, void **Cs,
                       int64_t M, int64_t K, int64_t N, int64_t n_batch,
                       Status *status) {
  *status = STATUS_SUCCESS;
  switch (dtype.code) {
    case TYPE_INT8: {
      BATCH_MM_KERNEL(int8_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_UINT8: {
      BATCH_MM_KERNEL(uint8_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_INT16: {
      BATCH_MM_KERNEL(int16_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_UINT16: {
      BATCH_MM_KERNEL(uint16_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_INT32: {
      BATCH_MM_KERNEL(int32_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_UINT32: {
      BATCH_MM_KERNEL(uint32_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_INT64: {
      BATCH_MM_KERNEL(int64_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_UINT64: {
      BATCH_MM_KERNEL(uint64_t, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_FLOAT: {
      BATCH_MM_KERNEL(float, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    case TYPE_DOUBLE: {
      BATCH_MM_KERNEL(double, As, Bs, Cs, M, K, N, n_batch);
      break;
    }
    default:
      *status = STATUS_NOT_SUPPORTED;
  }
}

/**
 * @brief Definition of aitisa_matmul.
 * 
 * @param tensor1 
 * @param tensor2 
 * @param output 
 * @return Status 
 */
Status aitisa_matmul(const Tensor tensor1, const Tensor tensor2,
                     Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  Status status;
  if (ndim_tensor1 == 1 && ndim_tensor2 == 1) {
    // vector-vector
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    if (dim0_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = 1;
    int64_t dims_out[1] = {1};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    vv_template(aitisa_tensor_data_type(tensor1), aitisa_tensor_data(tensor1),
                aitisa_tensor_data(tensor2), aitisa_tensor_data(*output),
                dim0_tensor1, &status);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 1) {
    // matrix-vector
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = 1;
    int64_t dims_out[1] = {dim0_tensor1};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    mv_template(aitisa_tensor_data_type(tensor1), aitisa_tensor_data(tensor1),
                aitisa_tensor_data(tensor2), aitisa_tensor_data(*output),
                dim0_tensor1, dim1_tensor1, &status);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 2) {
    // matrix-matrix
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    int64_t dim1_tensor2 = aitisa_tensor_dim(tensor2, 1);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = 2;
    int64_t dims_out[2] = {dim0_tensor1, dim1_tensor2};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    mm_template(aitisa_tensor_data_type(tensor1), aitisa_tensor_data(tensor1),
                aitisa_tensor_data(tensor2), aitisa_tensor_data(*output),
                dim0_tensor1, dim1_tensor1, dim1_tensor2, &status);

  } else if (ndim_tensor1 == 1 && ndim_tensor2 == 2) {
    // consider tensor1 as a matrix whose dim[0]=1;
    int64_t dim0_tensor1 = 1;
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    int64_t dim1_tensor2 = aitisa_tensor_dim(tensor2, 1);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = 1;
    int64_t dims_out[1] = {dim1_tensor2};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    mm_template(aitisa_tensor_data_type(tensor1), aitisa_tensor_data(tensor1),
                aitisa_tensor_data(tensor2), aitisa_tensor_data(*output),
                dim0_tensor1, dim1_tensor1, dim1_tensor2, &status);
  } else if (ndim_tensor1 >= 3 && ndim_tensor2 == 1) {
    // consider tensor2 as a matrix whose dim[1]=1;
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, ndim_tensor1 - 2);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, ndim_tensor1 - 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    int64_t dim1_tensor2 = 1;
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = ndim_tensor1 - 1;
    int64_t *dims_out =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(int64_t) * ndim_out);
    for (int64_t i = 0; i < ndim_out; ++i) {
      dims_out[i] = aitisa_tensor_dim(tensor1, i);
    }
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // data_ptr of batches
    int64_t n_batch = size_to_dim(ndim_tensor1 - 2, aitisa_tensor_dims(tensor1),
                                  ndim_tensor1);
    void **As =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void **Bs =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void **Cs =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void *data_tensor1 = aitisa_tensor_data(tensor1);
    void *data_tensor2 = aitisa_tensor_data(tensor2);
    void *data_output = aitisa_tensor_data(*output);
    int64_t size_mat1 = dim0_tensor1 * dim1_tensor1;
    int64_t size_mat3 = dim0_tensor1 * dim1_tensor2;
    int64_t size_type = aitisa_tensor_data_type(tensor1).size;
    for (int64_t i = 0; i < n_batch; ++i) {
      As[i] = data_tensor1 + i * size_mat1 * size_type;
      Bs[i] = data_tensor2;
      Cs[i] = data_output + i * size_mat3 * size_type;
    }
    // call kernel
    batch_mm_template(aitisa_tensor_data_type(tensor1), As, Bs, Cs,
                      dim0_tensor1, dim1_tensor1, dim1_tensor2, n_batch,
                      &status);
    // free memory
    aitisa_default_cpu_allocator()->raw_dealloc(Cs);
    aitisa_default_cpu_allocator()->raw_dealloc(Bs);
    aitisa_default_cpu_allocator()->raw_dealloc(As);
    aitisa_default_cpu_allocator()->raw_dealloc(dims_out);
  } else if (ndim_tensor1 == 1 && ndim_tensor2 >= 3) {
    // consider tensor1 as a matrix whose dim[0]=1;
    int64_t dim0_tensor1 = 1;
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, ndim_tensor2 - 2);
    int64_t dim1_tensor2 = aitisa_tensor_dim(tensor2, ndim_tensor2 - 1);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = ndim_tensor2 - 1;
    int64_t *dims_out =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(int64_t) * ndim_out);
    for (int64_t i = 0; i < ndim_out - 1; ++i) {
      dims_out[i] = aitisa_tensor_dim(tensor2, i);
    }
    dims_out[ndim_out - 1] = aitisa_tensor_dim(tensor2, ndim_tensor2 - 1);
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // data_ptr of batches
    int64_t n_batch = size_to_dim(ndim_tensor2 - 2, aitisa_tensor_dims(tensor2),
                                  ndim_tensor2);
    void **As =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void **Bs =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void **Cs =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void *data_tensor1 = aitisa_tensor_data(tensor1);
    void *data_tensor2 = aitisa_tensor_data(tensor2);
    void *data_output = aitisa_tensor_data(*output);
    int64_t size_mat2 = dim0_tensor2 * dim1_tensor2;
    int64_t size_mat3 = dim0_tensor1 * dim1_tensor2;
    int64_t size_type = aitisa_tensor_data_type(tensor1).size;
    for (int64_t i = 0; i < n_batch; ++i) {
      As[i] = data_tensor1;
      Bs[i] = data_tensor2 + i * size_mat2 * size_type;
      Cs[i] = data_output + i * size_mat3 * size_type;
    }
    // call kernel
    batch_mm_template(aitisa_tensor_data_type(tensor1), As, Bs, Cs,
                      dim0_tensor1, dim1_tensor1, dim1_tensor2, n_batch,
                      &status);
    // free memory
    aitisa_default_cpu_allocator()->raw_dealloc(Cs);
    aitisa_default_cpu_allocator()->raw_dealloc(Bs);
    aitisa_default_cpu_allocator()->raw_dealloc(As);
    aitisa_default_cpu_allocator()->raw_dealloc(dims_out);
  } else if (ndim_tensor1 >= 3 && ndim_tensor2 >= 2 ||
             ndim_tensor1 >= 2 && ndim_tensor2 >= 3) {
    // tensor-tensor => batch matrix-matrix
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, ndim_tensor1 - 2);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, ndim_tensor1 - 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, ndim_tensor2 - 2);
    int64_t dim1_tensor2 = aitisa_tensor_dim(tensor2, ndim_tensor2 - 1);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_DIMENSIONS_MISMATCH;
    }
    // create output
    int64_t ndim_out = max_int64_(ndim_tensor1, ndim_tensor2);
    int64_t *dims_out =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(int64_t) * ndim_out);
    CHECK_STATUS(aitisa_broadcast_array(
        aitisa_tensor_dims(tensor1), max_int64_(ndim_tensor1 - 2, 0),
        aitisa_tensor_dims(tensor2), max_int64_(ndim_tensor2 - 2, 0), dims_out,
        ndim_out - 2));
    dims_out[ndim_out - 2] = dim0_tensor1;
    dims_out[ndim_out - 1] = dim1_tensor2;
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // strides of batches
    int64_t ndim_batch = ndim_out - 2;
    int64_t *strides_t1 = aitisa_default_cpu_allocator()->raw_alloc(
        sizeof(int64_t) * (ndim_batch));
    int64_t *strides_t2 = aitisa_default_cpu_allocator()->raw_alloc(
        sizeof(int64_t) * (ndim_batch));
    batch_strides(aitisa_tensor_dims(tensor1), ndim_tensor1 - 2, strides_t1,
                  ndim_batch);
    batch_strides(aitisa_tensor_dims(tensor2), ndim_tensor2 - 2, strides_t2,
                  ndim_batch);
    // data_ptr of batches
    int64_t n_batch = size_to_dim(ndim_batch, dims_out, ndim_out);
    void **As =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void **Bs =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void **Cs =
        aitisa_default_cpu_allocator()->raw_alloc(sizeof(void *) * n_batch);
    void *data_tensor1 = aitisa_tensor_data(tensor1);
    void *data_tensor2 = aitisa_tensor_data(tensor2);
    void *data_output = aitisa_tensor_data(*output);
    int64_t size_mat1 = dim0_tensor1 * dim1_tensor1;
    int64_t size_mat2 = dim0_tensor2 * dim1_tensor2;
    int64_t size_mat3 = dim0_tensor1 * dim1_tensor2;
    int64_t size_type = aitisa_tensor_data_type(tensor1).size;
    for (int64_t i = 0; i < n_batch; ++i) {
      // linear index => offset
      int64_t idx = i;
      int64_t offset_t1 = 0, offset_t2 = 0;
      for (int j = 0; j < ndim_batch; ++j) {
        int64_t dim = dims_out[ndim_batch - 1 - j];
        int64_t rem = idx % dim;
        offset_t1 += rem * strides_t1[ndim_batch - 1 - j];
        offset_t2 += rem * strides_t2[ndim_batch - 1 - j];
        idx /= dim;
        if (idx == 0) {
          break;
        }
      }
      As[i] = data_tensor1 + offset_t1 * size_mat1 * size_type;
      Bs[i] = data_tensor2 + offset_t2 * size_mat2 * size_type;
      Cs[i] = data_output + i * size_mat3 * size_type;
    }
    // call kernel
    batch_mm_template(aitisa_tensor_data_type(tensor1), As, Bs, Cs,
                      dim0_tensor1, dim1_tensor1, dim1_tensor2, n_batch,
                      &status);
    // free memory
    aitisa_default_cpu_allocator()->raw_dealloc(Cs);
    aitisa_default_cpu_allocator()->raw_dealloc(Bs);
    aitisa_default_cpu_allocator()->raw_dealloc(As);
    aitisa_default_cpu_allocator()->raw_dealloc(strides_t2);
    aitisa_default_cpu_allocator()->raw_dealloc(strides_t1);
    aitisa_default_cpu_allocator()->raw_dealloc(dims_out);
  } else {
    status = STATUS_INVALID_ARGUMENT;
  }
  return status;
}
