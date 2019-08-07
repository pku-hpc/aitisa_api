#include "shape.h"

void create_shape(Layout layout, int64_t *dims, unsigned int ndim,
                  Shape *shape) {
  shape->ndim = ndim;
  shape->dims = (int64_t *)malloc(ndim * sizeof(int64_t));
  for(int i = 0;i < ndim;i++){
    shape->dims[i] = dims[i];
  }
  Layout new_layout;
  new_layout.type = layout.type;
  shape->layout = &new_layout;
}

void destroy_shape(Shape *shape){
  if (!shape) {
    return;
  }
  free(shape->dims);
  free(shape->layout);
  free(shape);
}
