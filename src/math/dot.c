#include "src/math/dot.h"
#include <stdlib.h>
#include "src/basic/slice.h"
#include "src/math/binary_op.h"
#include "src/math/matmul.h"
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

#define dot_kernel(typename, data1, data2, size, data_out)                            \
  ((typename *)data_out)[0] = 0;                                                      \
  for(int i=0; i<size; i++){                                                          \
    ((typename *)data_out)[0] += ((typename *)data1)[i] * ((typename *)data2)[i];     \
  }

static Status dot_template(DataType dtype, void* data1, void* data2,
                           int64_t size, void* data_out){
  Status status = STATUS_SUCCESS;
  switch (dtype.code) {
  case TYPE_INT8: {
    dot_kernel(int8_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_UINT8: {
    dot_kernel(uint8_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_INT16: {
    dot_kernel(int16_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_UINT16: {
    dot_kernel(uint16_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_INT32: {
    dot_kernel(int32_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_UINT32: {
    dot_kernel(uint32_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_INT64: {
    dot_kernel(int64_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_UINT64: {
    dot_kernel(uint64_t, data1, data2, size, data_out);
    break;
  }
  case TYPE_FLOAT: {
    dot_kernel(float, data1, data2, size, data_out);
    break;
  }
  case TYPE_DOUBLE: {
    dot_kernel(double, data1, data2, size, data_out);
    break;
  }
  default:
    status = STATUS_NOT_SUPPORTED;
  }
  return status;
}

#define dot_get_value_kernel(typename, tensor, idx, value)          \
  typename val = ((typename *)aitisa_tensor_data(tensor))[idx];     \
  *value = (double)val;

static Status dot_get_value(const Tensor tensor, int idx, double *value){
  DataType dtype = aitisa_tensor_data_type(tensor);
  switch (dtype.code) {
    case TYPE_INT8: {
      dot_get_value_kernel(int8_t, tensor, idx, value);
      break;
    }
    case TYPE_UINT8: {
      dot_get_value_kernel(uint8_t, tensor, idx, value);
      break;
    }
    case TYPE_INT16: {
      dot_get_value_kernel(int16_t, tensor, idx, value);
      break;
    }
    case TYPE_UINT16: {
      dot_get_value_kernel(uint16_t, tensor, idx, value);
      break;
    }
    case TYPE_INT32: {
      dot_get_value_kernel(int32_t, tensor, idx, value);
      break;
    }
    case TYPE_UINT32: {
      dot_get_value_kernel(uint32_t, tensor, idx, value);
      break;
    }
    case TYPE_INT64: {
      dot_get_value_kernel(int64_t, tensor, idx, value);
      break;
    }
    case TYPE_UINT64: {
      dot_get_value_kernel(uint64_t, tensor, idx, value);
      break;
    }
    case TYPE_FLOAT: {
      dot_get_value_kernel(float, tensor, idx, value);
      break;
    }
    case TYPE_DOUBLE: {
      dot_get_value_kernel(double, tensor, idx, value);
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }
  return STATUS_SUCCESS;
}

static Status scalar_tensor_mul(const Tensor tensor1, const Tensor tensor2,
                                Tensor *output){
  int64_t tensor1_ndim = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // create a new tensor full of the scalar value
  int64_t out_ndim;
  int64_t *out_dims;
  double value;
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(tensor1);
  Device device = aitisa_tensor_device(tensor1);
  if(tensor1_ndim > ndim_tensor2){
    out_ndim = tensor1_ndim;
    out_dims = aitisa_tensor_dims(tensor1);
    CHECK_STATUS(
      dot_get_value(tensor2, 0, &value));
  }else{
    out_ndim = ndim_tensor2;
    out_dims = aitisa_tensor_dims(tensor2);
    CHECK_STATUS(
      dot_get_value(tensor1, 0, &value));
  }
  CHECK_STATUS(
    aitisa_full(dtype, device, out_dims, out_ndim, value, &new_tensor));
  // implement tensor multiplication
  Status status;
  if(tensor1_ndim > ndim_tensor2){
    status = aitisa_mul(new_tensor, tensor1, output);
  }else{
    status = aitisa_mul(new_tensor, tensor2, output);
  }
  return status;
}

static Status vector_vector_dot(const Tensor tensor1, const Tensor tensor2,
                                Tensor *output){
  int64_t dim0_tensor1 = aitisa_tensor_dim(tensor1, 0);
  int64_t dim0_tensor2 = aitisa_tensor_dim(tensor2, 0);
  if (dim0_tensor1 != dim0_tensor2) {
    return STATUS_DIMENSIONS_MISMATCH;
  }
  // create output
  int64_t ndim_out = 1;
  int64_t dims_out[1] = {1};
  CHECK_STATUS(aitisa_full(aitisa_tensor_data_type(tensor1),
    aitisa_tensor_device(tensor1), dims_out, ndim_out,
    0.0, output));
  // call kernel
  Status status;
  status = dot_template(aitisa_tensor_data_type(tensor1), aitisa_tensor_data(tensor1),
                        aitisa_tensor_data(tensor2), tensor1->size,
                        aitisa_tensor_data(*output));
  return status;
}

static Status tensor_vector_dot(const Tensor tensor1, const Tensor tensor2,
                                Tensor *output){
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t *tensor1_dims = aitisa_tensor_dims(tensor1);
  int64_t *tensor2_dims = aitisa_tensor_dims(tensor2);
  // check whether the last dimension of tensor1 is equal to
  // the first dimension of tensor2
  if(tensor1_dims[ndim_tensor1-1] != tensor2_dims[0]){
    return STATUS_INVALID_ARGUMENT;
  }
  // create output
  int64_t out_ndim = ndim_tensor1 - 1;
  int64_t *out_dims =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_dims)*out_ndim);
  if(!out_dims){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<out_ndim; i++){
    out_dims[i] = tensor1_dims[i];
  }
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(tensor1);
  Device device = aitisa_tensor_device(tensor1);
  LayoutType layout_type = aitisa_tensor_layout_type(tensor1);
  CHECK_STATUS(
    aitisa_create(dtype, device, layout_type, out_dims, out_ndim, &new_tensor));
  *output = new_tensor;
  // make an index recorder of vectors in tensor1 then initialize it
  int *index_recorder =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder)*ndim_tensor1);
  if(!index_recorder){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim_tensor1; i++){
    index_recorder[i] = 0;
  }
  // make parameters for slice
    //size
  int *size =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*size)*ndim_tensor1);
  if(!size){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim_tensor1-1; i++){
    size[i] = 0;
  }
  size[ndim_tensor1-1] = tensor1_dims[ndim_tensor1-1];
    //step
  int *step =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*step)*ndim_tensor1);
  if(!step){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i< ndim_tensor1; i++){
    step[i] = 1;
  }
  // implement tensor-vector dot
  int64_t nvec_tensor1 = size_to_dim(ndim_tensor1-1, tensor1_dims, ndim_tensor1);
  Tensor vec2 = tensor2;
  Tensor vec1;
  Tensor result;
  void *out_data = aitisa_tensor_data(*output);
  int data_size = aitisa_tensor_data_type(*output).size;
  for(int vec_idx=0; vec_idx<nvec_tensor1; vec_idx++){
    CHECK_STATUS(
      aitisa_slice(tensor1, index_recorder, size, step, &vec1));
    CHECK_STATUS(
      vector_vector_dot(vec1, vec2, &result));
    // store the result into output
    void *result_data = aitisa_tensor_data(result);
    memcpy((char *)out_data+vec_idx*data_size, result_data, data_size);
    // update index_recorder
    for(int i=ndim_tensor1-2; i>=0; i--){
      index_recorder[i] += 1;
      /* judge whether the index is out of boundary */
      if(index_recorder[i] >= tensor1_dims[i]){
        index_recorder[i] = 0;
      }else{
        break;
      }
    }
  }
  aitisa_default_cpu_allocator()->raw_dealloc(index_recorder);
  aitisa_default_cpu_allocator()->raw_dealloc(size);
  aitisa_default_cpu_allocator()->raw_dealloc(step);
  return STATUS_SUCCESS;
}

static Status multidim_dot(const Tensor tensor1, const Tensor tensor2,
                           Tensor *output){
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  int64_t *tensor1_dims = aitisa_tensor_dims(tensor1);
  int64_t *tensor2_dims = aitisa_tensor_dims(tensor2);
  // check whether the last dimension of tensor1 is equal to
  // the first dimension of tensor2
  if(tensor1_dims[ndim_tensor1-1] != tensor2_dims[0]){
    return STATUS_INVALID_ARGUMENT;
  }
  // create output
  int64_t out_ndim = ndim_tensor1 + ndim_tensor2 - 2;
  int64_t *out_dims =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_dims)*out_ndim);
  if(!out_dims){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim_tensor1-1; i++){
    out_dims[i] = tensor1_dims[i];
  }
  for(int64_t i=ndim_tensor1-1; i<out_ndim; i++){
    out_dims[i] = tensor2_dims[i-ndim_tensor1+2];
  }
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(tensor1);
  Device device = aitisa_tensor_device(tensor1);
  LayoutType layout_type = aitisa_tensor_layout_type(tensor1);
  CHECK_STATUS(
    aitisa_create(dtype, device, layout_type, out_dims, out_ndim, &new_tensor));
  *output = new_tensor;
  // make index recorders of vectors in tensor1 and tensor2 then initialize them
  int *index_recorder_t1 =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder_t1)*ndim_tensor1);
  int *index_recorder_t2 =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index_recorder_t2)*ndim_tensor2);
  if(!index_recorder_t1 || !index_recorder_t2){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim_tensor1; i++){
    index_recorder_t1[i] = 0;
  }
  for(int64_t i=0; i<ndim_tensor2; i++){
    index_recorder_t2[i] = 0;
  }
  // make parameters for slice
    //size
  int *size_t1 =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*size_t1)*ndim_tensor1);
  int *size_t2 =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*size_t2)*ndim_tensor2);
  if(!size_t1 || !size_t2){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim_tensor1-1; i++){
    size_t1[i] = 0;
  }
  size_t1[ndim_tensor1-1] = tensor1_dims[ndim_tensor1-1];
  size_t2[0] = tensor2_dims[0];
  for(int64_t i=1; i<ndim_tensor2; i++){
    size_t2[i] = 0;
  }
    //step
  int *step_t1 =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*step_t1)*ndim_tensor1);
  int *step_t2 =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*step_t2)*ndim_tensor2);
  if(!step_t1 || !step_t2){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i< ndim_tensor1; i++){
    step_t1[i] = 1;
  }
  for(int64_t i=0; i< ndim_tensor2; i++){
    step_t2[i] = 1;
  }
  // implement multidim dot
  int64_t nvec_tensor1 = size_to_dim(ndim_tensor1-1, tensor1_dims, ndim_tensor1);
  int64_t nvec_tensor2 = size_from_dim(1, tensor2_dims, ndim_tensor2);
  Tensor vec2;
  Tensor vec1;
  Tensor result;
  void *out_data = aitisa_tensor_data(*output);
  int data_size = aitisa_tensor_data_type(*output).size;
  for(int vec1_idx=0; vec1_idx<nvec_tensor1; vec1_idx++){
    CHECK_STATUS(
      aitisa_slice(tensor1, index_recorder_t1, size_t1, step_t1, &vec1));
    for(int vec2_idx=0; vec2_idx<nvec_tensor2; vec2_idx++){
      CHECK_STATUS(
        aitisa_slice(tensor2, index_recorder_t2, size_t2, step_t2, &vec2));
      CHECK_STATUS(
        vector_vector_dot(vec1, vec2, &result));
      // store the result into output
      void *result_data = aitisa_tensor_data(result);
      memcpy((char*)out_data+(vec1_idx*nvec_tensor2+vec2_idx)*data_size, result_data, data_size);
      // update index_recorder_t2
      for(int i=ndim_tensor2-1; i>=1; i--){
        index_recorder_t2[i] += 1;
        /* judge whether the index is out of boundary */
        if(index_recorder_t2[i] >= tensor2_dims[i]){
          index_recorder_t2[i] = 0;
        }else{
          break;
        }
      }
    }
    // update index_recorder_t1
    for(int i=ndim_tensor1-2; i>=0; i--){
      index_recorder_t1[i] += 1;
      /* judge whether the index is out of boundary */
      if(index_recorder_t1[i] >= tensor1_dims[i]){
        index_recorder_t1[i] = 0;
      }else{
        break;
      }
    }
  }
  aitisa_default_cpu_allocator()->raw_dealloc(index_recorder_t1);
  aitisa_default_cpu_allocator()->raw_dealloc(size_t1);
  aitisa_default_cpu_allocator()->raw_dealloc(step_t1);
  aitisa_default_cpu_allocator()->raw_dealloc(index_recorder_t2);
  aitisa_default_cpu_allocator()->raw_dealloc(size_t2);
  aitisa_default_cpu_allocator()->raw_dealloc(step_t2);
  return STATUS_SUCCESS;
}

Status aitisa_dot(const Tensor tensor1, const Tensor tensor2,
                  Tensor *output){
  if(aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
    aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if(aitisa_tensor_data_type(tensor1).code !=
     aitisa_tensor_data_type(tensor2).code){
    return STATUS_NOT_SUPPORTED;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  Status status;
  if (ndim_tensor1 == 1 && ndim_tensor2 == 1) {
  // vector-vector, do the normal dot calculation
    status = vector_vector_dot(tensor1, tensor2, output);
  }else if(ndim_tensor1 == 2 && ndim_tensor2 == 2) {
  // matrix-matrix, do the matmul calculation
    status = aitisa_matmul(tensor1, tensor2, output);
  }else if(ndim_tensor1 == 0 || ndim_tensor2 == 0){
  // one of the two inputs is scalar, do the scalar-tensor multiplication
    status = scalar_tensor_mul(tensor1, tensor2, output);
  }else if(ndim_tensor2 == 1){
  // the second input is one-dimension, calculate the dot between the last
  // dimension of the first input and the second tensor
    status = tensor_vector_dot(tensor1, tensor2, output);
  }else if(ndim_tensor1 > 2 && ndim_tensor2 > 2){
  // FIXME2: This is not consistent with the definition when ndim_tensor1 > 2 && ndim_tensor2 > 2, 
  // check (https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html?highlight=dot#numpy.dot) for details.
  
  // both the dimensions of the two input are larger than 2, then calculate
  // the dot between the last dimension of the first input and the first
  // dimension of the second input
    status = multidim_dot(tensor1, tensor2, output);
  }else{
    status = STATUS_INVALID_ARGUMENT;
  }
  return status;
}