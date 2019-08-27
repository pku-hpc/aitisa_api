#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <string.h>
#include "src/core/shape.h"
#include "src/core/storage.h"
#include "src/core/status.h"
#include "src/core/macros.h"
#include "src/core/types.h"

/**
 * @brief Attributes of tensor struct
 * 
 * @detail Tensor structure contains all attributes which need to be known in a specific tensor
 */
struct _TensorImpl {
  int64_t offset; /* the storage offset when visit the tensor, useful in slice, split and so on */
  int64_t size; /* the total number of elements in this tensor */
  Shape shape; /* the dimension information of a tensor */
  Storage storage; /* the actual place where put the data elements */
};

typedef struct _TensorImpl* Tensor;

/**
 * @brief Get the number of dimensions of tensor
 */
static inline int64_t aitisa_tensor_ndim(const Tensor t) {
  return t->shape.ndim;
}

/**
 * @brief Get the number of elements in specific dimension of tensor
 */
static inline int64_t aitisa_tensor_dim(const Tensor t, int64_t d) {
  return t->shape.dims[d];
}

/**
 * @brief Get the detail of all dimensions of tensor
 */
static inline int64_t* aitisa_tensor_dims(const Tensor t) {
  return t->shape.dims;
}

/**
 * @brief Get the layout message of tensor
 */
static inline LayoutType aitisa_tensor_layout_type(const Tensor t) {
  return t->shape.layout.type;
}

/**
 * @brief Get the shape message of tensor
 */
static inline Shape aitisa_tensor_shape(const Tensor t) {
  return t->shape;
}

/**
 * @brief Get the storage message of tensor
 */
static inline Storage aitisa_tensor_storage(const Tensor t) {
  return t->storage;
}

/**
 * @brief Get total number of elements in tensor
 */
static inline int64_t aitisa_tensor_size(const Tensor t) {
  return t->size;
}

/**
 * @brief Get the void data pointer of elements in tensor
 */
static inline void* aitisa_tensor_data(const Tensor t) {
  return t->storage->data;
}

/**
 * @brief Get data type of elements in tensor
 */
static inline DataType aitisa_tensor_data_type(const Tensor t) {
  return t->storage->dtype;
}

/**
 * @brief Get the device message of tensor
 */
static inline Device aitisa_tensor_device(const Tensor t) {
  return t->storage->device;
}

/**
 * @brief Set the specific value to the idx-th data in tensor
 * 
 * @param t The tensor to be set value
 * @param idx The index of element that will be changed value
 * @param The value to be passed in the tensor
 */
static inline void aitisa_tensor_set_item(const Tensor t, int64_t idx,
                                          void *value) {
  char *data = (char *)aitisa_tensor_data(t);
  char *ptr = data + idx * aitisa_tensor_data_type(t).size;
  memcpy(ptr, value, aitisa_tensor_data_type(t).size);
  //aisisa_set_typed_value_func(aitisa_tensor_data_type(t))((void *)ptr, value);
}

/**
 * @brief Create a new tensor using the specific parameters
 * 
 * @param dtype The data type of tensor
 * @param device The device to create tensor on
 * @param layout_type The layout message of the tensor
 * @param dims The dimension detail of this tensor
 * @param ndim Number of dimension of this tensor
 * @param output A new tensor to be created
 * 
 * @code
 * Tensor tensor;
 * DataType dtype = {TYPE_INT32, sizeof(int)};
 * Device device = {DEVICE_CPU, 0};
 * int64_t dims[3] = {2, 3, 4};
 * aitisa_create(dtype, device, LAYOUT_DENSE, dims, 3, &tensor);
 * 
 * @return 
 * @retval STATUS_SUCCESS Successfully create a new tensor
 * @retval STATUS_ALLOC_FAILED Failed when the tensor already exists
 */
AITISA_API_PUBLIC Status aitisa_create(DataType dtype, Device device,
                                       LayoutType layout_type, int64_t *dims,
                                       int64_t ndim, Tensor *output);

/**
 * @brief Destroy an exist tensor
 * 
 * @param input the tensor to be destroy
 * 
 * @return 
 * @retval STATUS_SUCCESS Successfully destroy a tensor
 */
AITISA_API_PUBLIC Status aitisa_destroy(Tensor *input);

//void duplicate(Tensor input, Tensor *output);

//void full(DataType dtype, Device device, int64_t *dims, int64_t ndim,
//          void *value, Tensor *output);

#endif
