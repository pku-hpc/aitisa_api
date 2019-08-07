#include "tensor.h"

void aitisa_create(DataType dtype, Layout layout, Device device, int64_t *dims,
                   unsigned int ndim, Tensor *output) {
  Shape *shape = (Shape *)malloc(sizeof(shape));
  create_shape(layout, dims, ndim, shape);
  output->size = 1;
  for (int i = 0; i < ndim; i++) {
    output->size *= dims[i];
  }
  Storage *storage = (Storage *)malloc(sizeof(storage));
  create_storage(dtype, device, output->size, storage);
  output->shape = shape;
  output->storage = storage;
  output->offset = 0;
}

void aitisa_duplicate(Tensor input, Tensor *output) {
  aitisa_create(*input.storage->dtype, *input.shape->layout,
                *input.storage->device, input.shape->dims, input.shape->ndim,
                output);
}

void aitisa_destroy(Tensor *input) {
  if (!input) {
    return;
  }
  if (input->shape) {
    destroy_shape(input->shape);
  }
  if (input->storage) {
    destroy_storage(input->storage);
  }
  free(input);
}

void aitisa_full(DataType dtype, Device device, int64_t *dims,
                 unsigned int ndim, void *value, Tensor *output) {
  Shape *shape = (Shape *)malloc(sizeof(shape));
  Layout layout = {kDense};
  create_shape(layout, dims, ndim, shape);
  output->size = 1;
  for (int i = 0; i < ndim; i++) {
    output->size *= dims[i];
  }
  Storage *storage = (Storage *)malloc(sizeof(storage));
  create_storage(dtype, device, output->size, storage);
  switch (dtype.code) {
    case 0: {
      int8_t *data = (int8_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((int8_t *)value);
      }
      break;
    }
    case 1: {
      int16_t *data = (int16_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((int16_t *)value);
      }
      break;
    }
    case 2: {
      int32_t *data = (int32_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((int32_t *)value);
      }
      break;
    }
    case 3: {
      int64_t *data = (int64_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((int64_t *)value);
      }
      break;
    }
    case 4: {
      uint8_t *data = (uint8_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((uint8_t *)value);
      }
      break;
    }
    case 5: {
      uint16_t *data = (uint16_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((uint16_t *)value);
      }
      break;
    }
    case 6: {
      uint32_t *data = (uint32_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((uint32_t *)value);
      }
      break;
    }
    case 7: {
      uint64_t *data = (uint64_t *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((uint64_t *)value);
      }
      break;
    }
    case 8: {
      float *data = (float *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((float *)value);
      }
      break;
    }
    case 9: {
      double *data = (double *)&(storage->data);
      for (int i = 0; i < output->size; i++) {
        data[i] = *((double *)value);
      }
      break;
    }
    default:
      break;
  }
  // scalar_t *data = (scalar_t *)&(storage->data);
  // for(int i = 0;i < output->size;i++){
  //   data[i] = *((scalar_t *)value);
  // }
  output->shape = shape;
  output->storage = storage;
  output->offset = 0;

  TEST(dtype, value);
}