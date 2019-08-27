#ifndef STORAGE_H
#define STORAGE_H

#include <stdint.h>
#include <stdlib.h>
#include "src/core/device.h"
#include "src/core/status.h"
#include "src/core/types.h"

/**
 * @brief Attributes of storage struct
 * 
 * @detail Storage structure contains all attributes which need to be known in a specific storage
 */
struct _StorageImpl {
  DataType dtype; /* The data type of all elements */
  Device device; /* The device type to put storage on */
  int64_t size; /* The total memory size of elements in the tensor */
  void *data; /* The actual data which put in this storage */
};

typedef struct _StorageImpl *Storage;

/**
 * @brief Create a new storage using the specific parameters
 * 
 * @param dtype The data type of storage
 * @param device The device to create storage on
 * @param size The total number of elements that will put in this storage
 * @param storage A new storage to be created
 * 
 * @code
 * Storage storage;
 * DataType dtype = {TYPE_INT32, sizeof(int)};
 * Device device = {DEVICE_CPU, 0};
 * aitisa_create_storage(dtype, device, 12, storage);
 * 
 * @return 
 * @retval STATUS_SUCCESS Successfully create a new storage
 * @retval STATUS_ALLOC_FAILED Failed when the storage already exists
 */
Status aitisa_create_storage(DataType dtype, Device device, int64_t size,
                             Storage *storage);

/**
 * @brief Destroy an exist storage
 * 
 * @param input the storage to be destroy
 * 
 * @return 
 * @retval STATUS_SUCCESS Successfully destroy a storage
 */
Status aitisa_destroy_storage(Storage *storage);

#endif
