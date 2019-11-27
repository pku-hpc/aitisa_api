#include "src/basic/slice.h"
#include <inttypes.h>
#include <math.h>
#include "src/core/allocator.h"
#include "src/core/tensor.h"

static Status slice_create_output(const Tensor input, int *begin, int *size,
                                  int *step, Tensor *output) {
  int64_t in_ndim = aitisa_tensor_ndim(input);
  // calculate the dimensions of output
  int *out_dims_temp = aitisa_default_cpu_allocator()->raw_alloc(
      sizeof(*out_dims_temp) * in_ndim);
  if (!out_dims_temp) {
    return STATUS_ALLOC_FAILED;
  }
  for (int i = 0; i < in_ndim; i++) {
    out_dims_temp[i] = (int)ceil((double)size[i] / (double)step[i]);
  }
  int64_t out_ndim = 0;
  for (int i = 0; i < in_ndim; i++) {
    if (out_dims_temp[i] > 0) {
      out_ndim++;
    }
  }
  int64_t *out_dims =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_dims) * out_ndim);
  int last_idx = 0;
  for (int i = 0; i < in_ndim; i++) {
    if (out_dims_temp[i] > 0) {
      out_dims[last_idx++] = out_dims_temp[i];
    }
  }
  aitisa_default_cpu_allocator()->raw_dealloc(out_dims_temp);
  // create output
  Status status;
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  LayoutType layout_type = aitisa_tensor_layout_type(input);
  status = aitisa_create(dtype, device, layout_type, out_dims, out_ndim,
                         &new_tensor);
  *output = new_tensor;
  return status;
}

static Status slice_check_parameters(const Tensor input, int *begin, int *size,
                                     int *step) {
  int64_t in_ndim = aitisa_tensor_ndim(input);
  int64_t *in_dims = aitisa_tensor_dims(input);
  for (int i = 0; i < in_ndim; i++) {
    if (begin[i] < 0 || begin[i] >= in_dims[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
    if (size[i] < 0 || begin[i] + size[i] > in_dims[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
    if (step[i] < 1) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  return STATUS_SUCCESS;
}

#define slice_kernel(typename)                                                \
  typename *in_data = (typename *)aitisa_tensor_data(input);                  \
  typename *out_data = (typename *)aitisa_tensor_data(*output);               \
  int64_t out_size = aitisa_tensor_size(*output);                             \
  for (int out_linear_idx = 0; out_linear_idx < out_size; out_linear_idx++) { \
    /* get the linear index of input data element */                          \
    int64_t in_linear_idx = 0;                                                \
    for (int i = 0; i < in_ndim; i++) {                                       \
      in_linear_idx += index_recorder[i] * offset_recorder[i];                \
    }                                                                         \
    /* transport the input data element to output data */                     \
    out_data[out_linear_idx] = in_data[in_linear_idx];                        \
    /* update the index_recorder */                                           \
    for (int i = in_ndim - 1; i >= 0; i--) {                                  \
      index_recorder[i] += step[i];                                           \
      /* judge whether the index is out of boundary */                        \
      if (index_recorder[i] > boundary[i]) {                                  \
        index_recorder[i] = begin[i];                                         \
      } else {                                                                \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  }

static Status slice_template(const Tensor input, int *begin, int *size,
                             int *step, Tensor *output) {
  int64_t in_ndim = aitisa_tensor_ndim(input);
  int64_t *in_dims = aitisa_tensor_dims(input);
  /* make an index_recorder which records the index of present input
     element being processed, then initialize it*/
  int64_t *index_recorder = aitisa_default_cpu_allocator()->raw_alloc(
      sizeof(*index_recorder) * in_ndim);
  if (!index_recorder) {
    return STATUS_ALLOC_FAILED;
  }
  for (int i = 0; i < in_ndim; i++) {
    index_recorder[i] = begin[i];
  }
  /* make a boundary to judge whether the index is out of slice range*/
  int64_t *boundary =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*boundary) * in_ndim);
  if (!boundary) {
    return STATUS_ALLOC_FAILED;
  }
  for (int i = 0; i < in_ndim; i++) {
    boundary[i] = (int64_t)begin[i] + (int64_t)size[i] - 1;
  }
  /* make an offset_recorder which records every linear offset of each
   * dimension*/
  int64_t *offset_recorder = aitisa_default_cpu_allocator()->raw_alloc(
      sizeof(*offset_recorder) * in_ndim);
  if (!offset_recorder) {
    return STATUS_ALLOC_FAILED;
  }
  offset_recorder[in_ndim - 1] = 1;
  for (int i = in_ndim - 2; i >= 0; i--) {
    offset_recorder[i] = offset_recorder[i + 1] * in_dims[i + 1];
  }
  /* implement slice kernel*/
  DataType dtype = aitisa_tensor_data_type(input);
  switch (dtype.code) {
    case TYPE_INT8: {
      slice_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      slice_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      slice_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      slice_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      slice_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      slice_kernel(uint32_t);
      break;
    }
    case TYPE_INT64: {
      slice_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      slice_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      slice_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      slice_kernel(double);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }
  return STATUS_SUCCESS;
}

Status aitisa_slice(const Tensor input, int *begin, int *size, int *step,
                    Tensor *output) {
  // make sure parameters are correct
  CHECK_STATUS(slice_check_parameters(input, begin, size, step));
  // create output
  CHECK_STATUS(slice_create_output(input, begin, size, step, output));
  // implement slice
  Status status;
  status = slice_template(input, begin, size, step, output);
  return status;
}
