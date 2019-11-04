#include "src/basic/squeeze.h"
#include "src/core/allocator.h"
#include "src/basic/reshape.h"

Status aitisa_squeeze(const Tensor input, int64_t *axis,
                      int64_t num_axis, Tensor *output){
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t in_ndim = aitisa_tensor_ndim(input);
  // check if axis and num_axis are valid and
  // make the dimension to be deleted zero
  int64_t *processed_in_dims =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*processed_in_dims)*in_ndim);
  if(!processed_in_dims) return STATUS_ALLOC_FAILED;
  for(int64_t i=0; i<in_ndim; i++){
    processed_in_dims[i] = in_dims[i];
  }
  for(int64_t i=0; i<num_axis; i++){
    if(in_dims[axis[i]] != 1){
      return STATUS_INVALID_ARGUMENT;
    }
    processed_in_dims[axis[i]] = 0;
  }
  // create dims and ndim of output
  int64_t out_ndim = in_ndim - num_axis;
  int64_t *out_dims =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_dims)*out_ndim);
  if(!out_dims) return STATUS_ALLOC_FAILED;
  int64_t j = 0;
  for(int64_t i=0; i<in_ndim; i++){
    if(processed_in_dims[i] != 0){
      out_dims[j++] = processed_in_dims[i];
    }
  }
  // use reshape
  Status status =
    aitisa_reshape(input, out_dims, out_ndim, output);
  // destroy temporary parameters
  aitisa_default_cpu_allocator()->raw_dealloc(processed_in_dims);

  return status;
}
