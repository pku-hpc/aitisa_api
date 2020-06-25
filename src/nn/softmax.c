#include "src/core/tensor.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"
#include "src/nn/softmax.h"
#include <math.h>

static Status softmax_create_output(const Tensor input, Tensor *output){
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  // LayoutType layout_type = aitisa_tensor_layout_type(input);
  status = aitisa_create(dtype, device, dims, ndim, NULL, 0, &new_tensor);
  *output = new_tensor;
  return status;
}

#define softmax_with_all_kernel(typename, input, output)              \
  typename *in_data = (typename *)aitisa_tensor_data(input);          \
  typename *out_data = (typename *)aitisa_tensor_data(*output);       \
  /*calculate exp(in_data) and temporarily save them in out_data*/    \
  int64_t size = aitisa_tensor_size(input);                           \
  double total = 0;                                                   \
  for(int64_t i=0; i<size; i++){                                      \
    out_data[i] = (typename)exp(in_data[i]);                          \
    total += out_data[i];                                             \
  }                                                                   \
  /*calculate the real softmax value for output*/                     \
  for(int64_t i=0; i<size; i++){                                      \
    out_data[i] = (typename)(out_data[i] / total);                    \
  }

static Status softmax_with_all(const Tensor input, Tensor *output){
  // Choose the right data type
  DataType dtype = aitisa_tensor_data_type(input);
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, softmax_with_all_kernel, input, output);
  return STATUS_SUCCESS;
}

#define softmax_along_axis_kernel(typename)                     \
  typename *in_data =                                           \
    (typename *)aitisa_tensor_data(input);                      \
  typename *out_data =                                          \
    (typename *)aitisa_tensor_data(*output);                    \
  for(int64_t sample=0; sample<batch_size; sample++){           \
    /*get the linear index of the first element                 \
      of this sample*/                                          \
    int64_t linear_idx = 0;                                     \
    for(int64_t i=0; i<ndim; i++){                              \
      linear_idx +=                                             \
        index_recorder[i] * offset_recorder[i];                 \
    }                                                           \
    /*need a copy of linear _idx later*/                        \
    int64_t linear_idx_copy = linear_idx;                       \
    /*implement softmax*/                                       \
    double total = 0;                                           \
    for(int64_t i=0; i<dims[axis]; i++){                        \
      /*calculate the exp(in_data) and                          \
        temporarily save them in out_data*/                     \
      out_data[linear_idx] =                                    \
        (typename)exp(in_data[linear_idx]);                     \
      total += out_data[linear_idx];                            \
      /*update linear_idx*/                                     \
      linear_idx += element_offset;                             \
    }                                                           \
    linear_idx = linear_idx_copy;                               \
    for(int64_t i=0; i<dims[axis]; i++){                        \
      out_data[linear_idx] =                                    \
        (typename)(out_data[linear_idx] / total);               \
      /*update linear_idx*/                                     \
      linear_idx += element_offset;                             \
    }                                                           \
    /*update index_recorder*/                                   \
    for(int64_t i=ndim-1; i>=0; i--){                           \
      if(i == axis) continue;                                   \
      index_recorder[i] += 1;                                   \
      if(index_recorder[i] == dims[i]){                         \
        index_recorder[i] = 0;                                  \
      }else{                                                    \
        break;                                                  \
      }                                                         \
    }                                                           \
  }                                                             \



static Status softmax_along_axis(const Tensor input, const int axis,
                                 Tensor *output){
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  // Get the size of batch
  int64_t batch_size = 1;
  for(int64_t i=0; i<ndim; i++){
    if(i == axis){
      continue;
    }
    batch_size *= dims[i];
  }
  // Make an index-recorder of each sample
  int64_t *index_recorder =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder) * ndim);
  // Make an offset-recorder of each sample
  int64_t *offset_recorder =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder) * ndim);
  if(!index_recorder || !offset_recorder){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim; i++){
    index_recorder[i] = 0;
  }
  offset_recorder[ndim-1] = 1;
  for(int64_t i=ndim-2; i>=0; i--){
    offset_recorder[i] = dims[i+1] * offset_recorder[i+1];
  }
  // offset_recorder[axis] records the distance of
  // a pair of near element in the same sample
  int64_t element_offset = offset_recorder[axis];

  // Choose the right data type
  DataType dtype = aitisa_tensor_data_type(input);
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, softmax_along_axis_kernel);

  aitisa_default_cpu_allocator()->raw_dealloc(index_recorder);
  aitisa_default_cpu_allocator()->raw_dealloc(offset_recorder);
  return STATUS_SUCCESS;
}

Status aitisa_softmax(const Tensor input, const int axis,
                      Tensor *output){
  int64_t ndim = aitisa_tensor_ndim(input);
  if(axis < -1 || axis > ndim-1){
    return STATUS_NOT_SUPPORTED;
  }

  // Create output tensor
  CHECK_STATUS(softmax_create_output(input, output));

  Status status;
  switch(axis){
    case -1:{
      status = softmax_with_all(input, output);
      break;
    }
    default:{
      status = softmax_along_axis(input, axis, output);
    }
  }
  return status;
}
