#ifndef TENSOR_H_
#define TENSOR_H_

#include "shape.h"
#include "storage.h"

typedef struct Tensor {
  int64_t offset;
  int64_t size;
  Shape *shape;
  Storage *storage;
} Tensor;

void aitisa_create(DataType dtype, Layout layout, Device device, int64_t *dims, unsigned int ndim, Tensor *output);
void aitisa_full(DataType dtype, Device device, int64_t *dims, unsigned int ndim, void *value, Tensor *output);
void aitisa_duplicate(Tensor input, Tensor *output);
void aitisa_destroy(Tensor *input);


#endif
