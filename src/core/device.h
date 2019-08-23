#ifndef DEVICE_H
#define DEVICE_H

#include "src/core/macros.h"

typedef enum {
  DEVICE_CPU = 0,
  DEVICE_CUDA = 1
} DeviceType;

typedef struct {
  DeviceType type;
  int id;
} Device;

#endif
