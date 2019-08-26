#include "src/math/matmul.h"
#include "src/basic/broadcast.h"
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

int64_t max_int64_(int64_t a, int64_t b) { return a > b ? a : b; }

int64_t batch_strides(int64_t *dims, int64_t ndim, int64_t *strides,
                      int64_t nstride) {
  int64_t stride = 1;
  for (int64_t i = 0; i < nstride; ++i) {
    if (ndim - 1 - i >= 0 && dims[ndim - 1 - i] != 1) {
      strides[nstride - 1 - i] = stride;
      stride *= dims[ndim - 1 - i];
    } else {
      strides[nstride - 1 - i] = 0;
    }
  }
}

void vv(void *X, void *Y, void *Z, int64_t L) {
  for (int i = 0; i < L; ++i) {
    ((float *)Z)[0] += ((float *)X)[i] * ((float *)Y)[i];
  }
}

void mv(void *A, void *X, void *B, int64_t M, int64_t N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      ((float *)B)[i] += ((float *)A)[i * N + j] * ((float *)X)[j];
    }
  }
}

void mm(void *A, void *B, void *C, int64_t M, int64_t K, int64_t N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int q = 0; q < K; ++q) {
        ((float *)C)[i * N + j] +=
            ((float *)A)[i * K + q] * ((float *)B)[q * N + j];
      }
    }
  }
}

void batch_mm(void **As, void **Bs, void **Cs, int64_t M, int64_t K, int64_t N,
              int64_t n_batch) {
  for (int64_t bth = 0; bth < n_batch; ++bth) {
    float *A = *((float **)As + bth);
    float *B = *((float **)Bs + bth);
    float *C = *((float **)Cs + bth);
    mm(A, B, C, M, K, N);
  }
}

Status aitisa_matmul(const Tensor tensor1, const Tensor tensor2,
                     Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU ||
      aitisa_tensor_data_type(tensor1).code != TYPE_FLOAT) {
    return STATUS_NOT_SUPPORTED;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  Status status = STATUS_SUCCESS;
  if (ndim_tensor1 == 1 && ndim_tensor2 == 1) {
    // vector-vector
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    if (dim0_tensor1 != dim0_tensor2) {
      return STATUS_INVALID_ARGUMENT;
    }
    // create output
    int64_t ndim_out = 1;
    int64_t dims_out[1] = {1};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    vv(aitisa_tensor_data(tensor1), aitisa_tensor_data(tensor2),
       aitisa_tensor_data(*output), dim0_tensor1);

  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 1) {
    // matrix-vector
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_INVALID_ARGUMENT;
    }
    // create output
    int64_t ndim_out = 1;
    int64_t dims_out[1] = {dim0_tensor1};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    mv(aitisa_tensor_data(tensor1), aitisa_tensor_data(tensor2),
       aitisa_tensor_data(*output), dim0_tensor1, dim1_tensor1);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 2) {
    // matrix-matrix
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    int64_t dim1_tensor2 = aitisa_tensor_dim(tensor2, 1);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_INVALID_ARGUMENT;
    }
    // create output
    int64_t ndim_out = 2;
    int64_t dims_out[2] = {dim0_tensor1, dim1_tensor2};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    mm(aitisa_tensor_data(tensor1), aitisa_tensor_data(tensor2),
       aitisa_tensor_data(*output), dim0_tensor1, dim1_tensor1, dim1_tensor2);
  } else if (ndim_tensor1 == 1 && ndim_tensor2 == 2) {
    // consider tensor1 as a matrix whose dim[0]=1;
    int64_t dim0_tensor1 = 1;
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, 0);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    int64_t dim1_tensor2 = aitisa_tensor_dim(tensor2, 1);
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_INVALID_ARGUMENT;
    }
    // create output
    int64_t ndim_out = 1;
    int64_t dims_out[1] = {dim1_tensor2};
    CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1), dims_out, ndim_out,
                             0.0, output));
    // call kernel
    mm(aitisa_tensor_data(tensor1), aitisa_tensor_data(tensor2),
       aitisa_tensor_data(*output), dim0_tensor1, dim1_tensor1, dim1_tensor2);
  } else if (ndim_tensor1 >= 3 && ndim_tensor2 == 1) {
    // consider tensor2 as a matrix whose dim[1]=1;
    int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, ndim_tensor1 - 2);
    int64_t dim1_tensor1 = aitisa_tensor_dim(tensor1, ndim_tensor1 - 1);
    int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
    int64_t dim1_tensor2 = 1;
    if (dim1_tensor1 != dim0_tensor2) {
      return STATUS_INVALID_ARGUMENT;
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
    batch_mm(As, Bs, Cs, dim0_tensor1, dim1_tensor1, dim1_tensor2, n_batch);
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
      return STATUS_INVALID_ARGUMENT;
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
    batch_mm(As, Bs, Cs, dim0_tensor1, dim1_tensor1, dim1_tensor2, n_batch);
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
      return STATUS_INVALID_ARGUMENT;
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
    batch_mm(As, Bs, Cs, dim0_tensor1, dim1_tensor1, dim1_tensor2, n_batch);
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
