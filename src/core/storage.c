#include "storage.h"
void create_storage(DataType dtype, Device device, int64_t size,
                    Storage *storage) {
  Device new_device = {device.device_id, device.device_type};
  DataType new_dtype = {dtype.code, dtype.bits};
  storage->device = &new_device;
  storage->dtype = &new_dtype;
  storage->size = size * dtype.bits;
  storage->data = malloc(storage->size);
}

void destroy_storage(Storage *storage) {
  if (!storage) {
    return;
  }
  free(storage->device);
  free(storage->dtype);
  free(storage->data);
  free(storage);
}