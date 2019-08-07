#ifndef STORAGE_H_
#define STORAGE_H_

#include <stdint.h>
#include <stdlib.h>
#include "allocator.h"
#include "device.h"

typedef enum {
  kInt8 = 0U,
  kInt16 = 1U,
  kInt32 = 2U,
  kInt64 = 3U,
  kUInt8 = 4U,
  kUInt16 = 5U,
  kUInt32 = 6U,
  kUInt64 = 7U,
  kFloat = 8U,
  kDouble = 9U
} ScalarType;

typedef struct DataType {
  ScalarType code;
  uint8_t bits;
} DataType;

// size重名
typedef struct Storage {
  void *data;
  int64_t size;
  DataType *dtype;
  Device *device;
  // Allocator *allocator;
} Storage;

void create_storage(DataType dtype, Device device, int64_t size,
                    Storage *storage);

void destroy_storage(Storage *storage);

#endif
