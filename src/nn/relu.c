#include "src/nn/relu.h"

#define relu_kernel(typename)                       \
  typename *in_data = aitisa_tensor_data(input);    \
  typename *out_data = aitisa_tensor_data(*output); \
  for (int64_t i = 0; i < size; i++) {              \
    if (in_data[i] > 0) {                           \
      out_data[i] = in_data[i];                     \
    } else {                                        \
      out_data[i] = 0;                              \
    }                                               \
  }

Status aitisa_relu(const Tensor input, Tensor *output) {
  // Create output
  int64_t *dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  LayoutType layout_type = aitisa_tensor_layout_type(input);
  CHECK_STATUS(
      aitisa_create(dtype, device, layout_type, dims, ndim, &new_tensor));
  *output = new_tensor;
  // Implement relu
  int64_t size = aitisa_tensor_size(input);
  Status status = STATUS_SUCCESS;
  switch (dtype.code) {
    case TYPE_INT8: {
      relu_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      relu_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      relu_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      relu_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      relu_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      relu_kernel(uint32_t);
      break;
    }
    case TYPE_INT64: {
      relu_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      relu_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      relu_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      relu_kernel(double);
      break;
    }
    default:
      status = STATUS_NOT_SUPPORTED;
  }
  return status;
}
