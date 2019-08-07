#ifndef SHAPE_H_
#define SHAPE_H_

#include <stdint.h>
#include <stdlib.h>

typedef enum { 
  kDense = 0,
  kSparse = 1,
  kInvalid = 2
} LayoutType;

// Tensorflow max_sparse_elements
typedef struct Layout {
  LayoutType type;
  // int32_t *minor_to_major;
} Layout;

typedef struct Shape {
  int64_t *dims;
  int64_t ndim;
  Layout *layout;
} Shape;

void create_shape(Layout layout, int64_t *dims, unsigned int ndim, Shape *shape);
void destroy_shape(Shape *shape);

#endif
