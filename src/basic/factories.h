#ifndef FACTORIES_H
#define FACTORIES_H

#include "src/core/tensor.h"

AITISA_API_PUBLIC Status aitisa_full(DataType dtype, Device device,
                                     int64_t *dims, int64_t ndim, double value,
                                     Tensor *output);
#endif