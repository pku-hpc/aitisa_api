#ifndef SHAPE_H
#define SHAPE_H

#include <stdint.h>
#include <stdlib.h>

#include "src/core/macros.h"
#include "src/core/status.h"

typedef enum {
  LAYOUT_DENSE = 0,
  LAYOUT_SPARSE = 1,
} LayoutType;

typedef struct {
  LayoutType type;
  int64_t *min2maj;
} Layout;

typedef struct {
  int64_t ndim;
  int64_t *dims;
  Layout layout;
} Shape;

Status aitisa_create_shape(LayoutType layout_type, int64_t *dims,
                           int64_t ndim, Shape *shape);

Status aitisa_destroy_shape(Shape *shape);

#endif
