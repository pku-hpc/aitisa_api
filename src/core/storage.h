#ifndef STORAGE_H
#define STORAGE_H

#include <stdint.h>
#include <stdlib.h>
#include "src/core/device.h"
#include "src/core/status.h"
#include "src/core/types.h"

/* The implementation of the storage hold by the tensor */
struct _StorageImpl {
  /* The data type of the elements in the storage */
  DataType dtype;
  /* The device where the elements are stored */
  Device device;
  /* The number of the elements hold by the storage */
  int64_t size;
  /* The raw data pointer pointing to the memory space */
  void *data;
};

/* The storage handle for users without exposing the implementation details. */
typedef struct _StorageImpl *Storage;

/*
 * Create a new storage based on the data type, the device, the size 
 * of elements.
 */
Status aitisa_create_storage(DataType dtype, Device device, int64_t size,
                             Storage *storage);

/* Destory the storage including all its members */
Status aitisa_destroy_storage(Storage *storage);

#endif
