#include "src/math/sqrt.h"
#include <math.h>

static Status sqrt_create_output(const Tensor input, Tensor *output) {
  Status status;
  int64_t *dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  LayoutType layout_type = aitisa_tensor_layout_type(input);
  status = aitisa_create(dtype, device, layout_type, dims, ndim, &new_tensor);
  if (status == STATUS_SUCCESS) {
    *output = new_tensor;
  }
  return status;
}

#define sqrt_kernel(typename, input, output)                    \
  typename *in_data = (typename *)aitisa_tensor_data(input);    \
  typename *out_data = (typename *)aitisa_tensor_data(*output); \
  for (int64_t i = 0; i < size; i++) {                          \
    if (in_data[i] < 0) return STATUS_MATH_ERROR;               \
    out_data[i] = (typename)sqrt(in_data[i]);                   \
  }

static Status sqrt_template(const Tensor input, Tensor *output) {
  Status status = STATUS_SUCCESS;
  int64_t size = aitisa_tensor_size(input);
  DataType dtype = aitisa_tensor_data_type(input);
  switch (dtype.code) {
    case TYPE_INT8: {
      sqrt_kernel(int8_t, input, output);
      break;
    }
    case TYPE_UINT8: {
      sqrt_kernel(uint8_t, input, output);
      break;
    }
    case TYPE_INT16: {
      sqrt_kernel(int16_t, input, output);
      break;
    }
    case TYPE_UINT16: {
      sqrt_kernel(uint16_t, input, output);
      break;
    }
    case TYPE_INT32: {
      sqrt_kernel(int32_t, input, output);
      break;
    }
    case TYPE_UINT32: {
      sqrt_kernel(uint32_t, input, output);
      break;
    }
    case TYPE_INT64: {
      sqrt_kernel(int64_t, input, output);
      break;
    }
    case TYPE_UINT64: {
      sqrt_kernel(uint64_t, input, output);
      break;
    }
    case TYPE_FLOAT: {
      sqrt_kernel(float, input, output);
      break;
    }
    case TYPE_DOUBLE: {
      sqrt_kernel(double, input, output);
      break;
    }
    default:
      status = STATUS_NOT_SUPPORTED;
  }
  return status;
}

Status aitisa_sqrt(const Tensor input, Tensor *output) {
  // Create output
  CHECK_STATUS(sqrt_create_output(input, output));
  // Implement square root
  Status status = sqrt_template(input, output);
  return status;
}
