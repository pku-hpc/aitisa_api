#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <string.h>
#include "src/core/shape.h"
#include "src/core/storage.h"
#include "src/core/status.h"
#include "src/core/macros.h"
#include "src/core/types.h"

struct _TensorImpl {
  int64_t offset;
  int64_t size;
  Shape shape;
  Storage storage;
};

typedef struct _TensorImpl* Tensor;

static inline int64_t aitisa_tensor_ndim(const Tensor t) {
  return t->shape.ndim;
}

static inline int64_t aitisa_tensor_dim(const Tensor t, int64_t d) {
  return t->shape.dims[d];
}

static inline int64_t* aitisa_tensor_dims(const Tensor t) {
  return t->shape.dims;
}

static inline LayoutType aitisa_tensor_layout_type(const Tensor t) {
  return t->shape.layout.type;
}

static inline Shape aitisa_tensor_shape(const Tensor t) {
  return t->shape;
}

static inline Storage aitisa_tensor_storage(const Tensor t) {
  return t->storage;
}

static inline int64_t aitisa_tensor_size(const Tensor t) {
  return t->size;
}

static inline void* aitisa_tensor_data(const Tensor t) {
  return t->storage->data;
}

static inline DataType aitisa_tensor_data_type(const Tensor t) {
  return t->storage->dtype;
}
static inline Device aitisa_tensor_device(const Tensor t) {
  return t->storage->device;
}

static inline void aitisa_tensor_set_item(const Tensor t, int64_t idx,
                                          void *value) {
  char *data = (char *)aitisa_tensor_data(t);
  char *ptr = data + idx * aitisa_tensor_data_type(t).size;
  memcpy(ptr, value, aitisa_tensor_data_type(t).size);
  //aisisa_set_typed_value_func(aitisa_tensor_data_type(t))((void *)ptr, value);
}

AITISA_API_PUBLIC Status aitisa_create(DataType dtype, Device device,
                                       LayoutType layout_type, int64_t *dims,
                                       int64_t ndim, Tensor *output);

AITISA_API_PUBLIC Status aitisa_destroy(Tensor *input);

//void duplicate(Tensor input, Tensor *output);

//void full(DataType dtype, Device device, int64_t *dims, int64_t ndim,
//          void *value, Tensor *output);

#endif
