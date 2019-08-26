#ifndef INDEX_UTILS_H
#define INDEX_UTILS_H

#include <stdint.h>

#include "src/core/tensor.h"

AITISA_API_PUBLIC int64_t aitisa_get_stride(const Tensor t, int64_t dimension);
AITISA_API_PUBLIC void aitisa_get_all_strides(const Tensor t, int64_t *strides);

AITISA_API_PUBLIC int64_t aitisa_coords_to_linidx(const Tensor t,
                                                  int64_t *coords, int64_t len);
AITISA_API_PUBLIC void aitisa_linidx_to_coords(const Tensor t, int64_t linidx,
                                               int64_t *coords);

AITISA_API_PUBLIC int64_t aitisa_coords_to_offset(const Tensor t,
                                                  int64_t *coords, int64_t len);
AITISA_API_PUBLIC void aitisa_offset_to_coords(const Tensor t, int64_t offset,
                                               int64_t *coords);

#endif