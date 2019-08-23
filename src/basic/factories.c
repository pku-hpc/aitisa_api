#include "src/basic/factories.h"

Status aitisa_full(DataType dtype, Device device, int64_t *dims, int64_t ndim,
                   double value, Tensor *output) {
  Tensor new_tensor;
  CHECK_STATUS(
      aitisa_create(dtype, device, LAYOUT_DENSE, dims, ndim, &new_tensor));
  int64_t size = aitisa_tensor_size(new_tensor);
  aisisa_castto_typed_value_func(dtype)(&value, &value);
  for (int i = 0; i < size; ++i) {
    aitisa_tensor_set_item(new_tensor, i, &value);
  }
  *output = new_tensor;
  return STATUS_SUCCESS;
}