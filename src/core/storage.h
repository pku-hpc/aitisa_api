#ifndef STORAGE_H
#define STORAGE_H

#include <stdint.h>
#include <stdlib.h>
#include "src/core/device.h"
#include "src/core/status.h"
#include "src/core/types.h"

struct _StorageImpl {
  DataType dtype;
  Device device;
  int64_t size;
  void *data;
};

typedef struct _StorageImpl *Storage;

Status aitisa_create_storage(DataType dtype, Device device, int64_t size,
                             Storage *storage);

Status aitisa_destroy_storage(Storage *storage);

#endif
