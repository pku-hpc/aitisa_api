#include "src/core/tensor.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

Status aitisa_create(DataType dtype, Device device, LayoutType layout_type,
            int64_t *dims, int64_t ndim, Tensor *output) {
  Tensor tensor;
  tensor = aitisa_default_cpu_allocator()->raw_alloc(sizeof(*tensor)); 
  if (!tensor) return STATUS_ALLOC_FAILED; 
  tensor->size = size_of_dims(dims, ndim);
  tensor->offset = 0;
  CHECK_STATUS(aitisa_create_shape(layout_type, dims, ndim, &tensor->shape));
  CHECK_STATUS(
      aitisa_create_storage(dtype, device, tensor->size, &tensor->storage));
  *output = tensor;
  return STATUS_SUCCESS;
}

Status aitisa_destroy(Tensor *input) {
  if (!(*input)) return STATUS_SUCCESS;
  CHECK_STATUS(aitisa_destroy_shape(&(*input)->shape));
  CHECK_STATUS(aitisa_destroy_storage(&(*input)->storage));
  aitisa_default_cpu_allocator()->raw_dealloc((*input));
  return STATUS_SUCCESS;
}

