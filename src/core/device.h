#ifndef DEVICE_H_
#define DEVICE_H_

typedef enum {
  kCPU = 0,
  kCUDA = 1
} DeviceType;

typedef struct {
  DeviceType device_type;
  int device_id;
} Device;

#endif
