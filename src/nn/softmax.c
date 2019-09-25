#include "src/core/tensor.h"
#include "src/core/allocator.h"
#include <math.h>

static Status softmax_create_output(const Tensor input, Tensor *output){
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  LayoutType layout_type = aitisa_tensor_layout_type(input);
  status = aitisa_create(dtype, device, layout_type, dims, ndim, &new_tensor);
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
  // choose the right data type
  DataType dtype = aitisa_tensor_data_type(input);
  switch(dtype.code){
    case TYPE_INT8: {
      softmax_with_all_kernel(int8_t, input, output);
      break;
    }
    case TYPE_UINT8: {
      softmax_with_all_kernel(uint8_t, input, output);
      break;
    }
    case TYPE_INT16: {
      softmax_with_all_kernel(int16_t, input, output);
      break;
    }
    case TYPE_UINT16: {
      softmax_with_all_kernel(uint16_t, input, output);
      break;
    }
    case TYPE_INT32: {
      softmax_with_all_kernel(int32_t, input, output);
      break;
    }
    case TYPE_UINT32: {
      softmax_with_all_kernel(uint32_t, input, output);
      break;
    }
    case TYPE_INT64: {
      softmax_with_all_kernel(int64_t, input, output);
      break;
    }
    case TYPE_UINT64: {
      softmax_with_all_kernel(uint64_t, input, output);
      break;
    }
    case TYPE_FLOAT: {
      softmax_with_all_kernel(float, input, output);
      break;
    }
    case TYPE_DOUBLE: {
      softmax_with_all_kernel(double, input, output);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
    }
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
  /*get the size of batch*/
  int64_t batch_size = 1;
  for(int64_t i=0; i<ndim; i++){
    if(i == axis){
      continue;
    }
    batch_size *= dims[i];
  }
  /*make an index-recorder of each sample*/
  int64_t *index_recorder =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder) * ndim);
  if(!index_recorder){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim; i++){
    index_recorder[i] = 0;
  }
  /*make an offset-recorder of each sample*/
  int64_t *offset_recorder =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder) * ndim);
  if(!offset_recorder){
    return STATUS_ALLOC_FAILED;
  }
  offset_recorder[ndim-1] = 1;
  for(int64_t i=ndim-2; i>=0; i--){
    offset_recorder[i] = dims[i+1] * offset_recorder[i+1];
  }
  /*offset_recorder[axis] records the distance of
    a pair of near element in the same sample*/
  int64_t element_offset = offset_recorder[axis];

  /* choose the right data type*/
  DataType dtype = aitisa_tensor_data_type(input);
  switch(dtype.code){
    case TYPE_INT8: {
      softmax_along_axis_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      softmax_along_axis_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      softmax_along_axis_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      softmax_along_axis_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      softmax_along_axis_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      softmax_along_axis_kernel(uint32_t);
      break;
    }
    case TYPE_INT64: {
      softmax_along_axis_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      softmax_along_axis_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      softmax_along_axis_kernel(float)
      break;
    }
    case TYPE_DOUBLE: {
      softmax_along_axis_kernel(double);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }
  return STATUS_SUCCESS;
}

Status aitisa_softmax(const Tensor input, const int axis,
                      Tensor *output){
  Status status;
  int64_t ndim = aitisa_tensor_ndim(input);
  if(axis < -1 || axis > ndim-1){
    status =  STATUS_NOT_SUPPORTED;
    return status;
  }

  // create output tensor
  CHECK_STATUS(
    softmax_create_output(input, output));

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
