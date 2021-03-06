#include "src/nn/sigmoid.h"
#include "src/core/dispatch.h"
#include <math.h>

#define sigmoid_kernel(typename)                          \
  typename *in_data = aitisa_tensor_data(input);          \
  typename *out_data = aitisa_tensor_data(*output);       \
  for (int64_t i = 0; i < size; i++) {                    \
    out_data[i] = (typename)(1 / (1 + exp(-in_data[i]))); \
  }

Status aitisa_sigmoid(const Tensor input, Tensor *output) {
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
  // Implement sigmoid
  Status status = STATUS_SUCCESS;
  int64_t size = aitisa_tensor_size(input);
  AITISA_DISPATCH_FLOATING_TYPES_RETURN(dtype, sigmoid_kernel);
  return status;
}
