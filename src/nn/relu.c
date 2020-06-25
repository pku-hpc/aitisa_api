#include "src/nn/relu.h"
#include "src/core/dispatch.h"

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
  // LayoutType layout_type = aitisa_tensor_layout_type(input);
  CHECK_STATUS(
      aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor));
  *output = new_tensor;
  // Implement relu
  int64_t size = aitisa_tensor_size(input);
  Status status = STATUS_SUCCESS;
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, relu_kernel);
  return status;
}
