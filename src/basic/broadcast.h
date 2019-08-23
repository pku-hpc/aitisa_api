#ifndef BROADCAST_H
#define BROADCAST_H

#include "src/core/tensor.h"

AITISA_API_PUBLIC Status
aitisa_broadcast_array(int64_t* dims_in1, int64_t ndim_in1, int64_t* dims_in2,
                       int64_t ndim_in2, int64_t* dims_out, int64_t ndim_out);

#endif