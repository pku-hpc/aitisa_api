#include "src/math/sqrt.h"
#include "src/core/dispatch.h"
#include <math.h>

static Status sqrt_create_output(const Tensor input, Tensor *output) {
  Status status;
  int64_t *dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  // LayoutType layout_type = aitisa_tensor_layout_type(input);
  status = aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor);
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
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, sqrt_kernel, input, output);
  return status;
}

Status aitisa_sqrt(const Tensor input, Tensor *output) {
  // Create output
  CHECK_STATUS(sqrt_create_output(input, output));
  // Implement square root
  Status status = sqrt_template(input, output);
  return status;
}
