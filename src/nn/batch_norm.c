#include "src/nn/batch_norm.h"
#include "src/basic/slice.h"
#include "src/basic/factories.h"
#include "src/core/utils.h"
#include "src/core/allocator.h"
#include "src/math/sqrt.h"
#include "src/math/binary_op.h"
#include "src/basic/squeeze.h"

static Status batch_norm_check_parameters(const Tensor input, const int axis,
                                   const Tensor scale, const Tensor bias,
                                   const Tensor mean, const Tensor variance,
                                   const double epsilon){
  // check input
  DataType in_dtype = aitisa_tensor_data_type(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  if(ndim<2 || ndim>5){
    return STATUS_DIMENSIONS_MISMATCH;
  }
  // check axis
  if(ndim == 2){
    if(axis != 1) return STATUS_INVALID_ARGUMENT;
  }else if(ndim>2 && ndim<6){
    if(axis!=1 && axis!=ndim-1){
      return STATUS_INVALID_ARGUMENT;
    }
  }else{
    return STATUS_DIMENSIONS_MISMATCH;
  }
  // check scale and bias
  DataType scale_dtype = aitisa_tensor_data_type(scale);
  if(scale_dtype.code != in_dtype.code) return STATUS_TYPE_MISMATCH;
  DataType bias_dtype = aitisa_tensor_data_type(bias);
  if(bias_dtype.code != in_dtype.code) return STATUS_TYPE_MISMATCH;
  int64_t num_channels = aitisa_tensor_dim(input, axis);
  int64_t scale_size = aitisa_tensor_size(scale);
  int64_t bias_size = aitisa_tensor_size(bias);
  if(scale_size!= num_channels || bias_size!=num_channels){
    return STATUS_DIMENSIONS_MISMATCH;
  }
  int64_t scale_ndim = aitisa_tensor_ndim(scale);
  int64_t bias_ndim = aitisa_tensor_ndim(bias);
  if(scale_ndim!=1 || bias_ndim!=1){
    return STATUS_DIMENSIONS_MISMATCH;
  }
  //check mean and variance
  DataType mean_dtype = aitisa_tensor_data_type(mean);
  if(mean_dtype.code != in_dtype.code) return STATUS_TYPE_MISMATCH;
  DataType var_dtype = aitisa_tensor_data_type(variance);
  if(var_dtype.code != in_dtype.code) return STATUS_TYPE_MISMATCH;
  int64_t mean_size = aitisa_tensor_size(mean);
  int64_t var_size = aitisa_tensor_size(variance);
  if(mean_size!= num_channels || var_size!=num_channels){
    return STATUS_DIMENSIONS_MISMATCH;
  }
  int64_t mean_ndim = aitisa_tensor_ndim(mean);
  int64_t var_ndim = aitisa_tensor_ndim(variance);
  if(mean_ndim!=1 || var_ndim!=1){
    return STATUS_DIMENSIONS_MISMATCH;
  }
  //check epsilon
  if(epsilon < 0){
    return STATUS_INVALID_ARGUMENT;
  }

  return STATUS_SUCCESS;
}

static Status batch_norm_create_output(const Tensor input, Tensor *output){
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  LayoutType layout_type = aitisa_tensor_layout_type(input);
  status =
    aitisa_create(dtype, device, layout_type, dims, ndim, &new_tensor);
  if(status == STATUS_SUCCESS){
    *output = new_tensor;
  }
  return status;
}

// This function would transport the data the source tensor
// to destination tensor
static void batch_norm_transport(const Tensor source, Tensor *destination,
                                   int64_t *index_recorder,
                                   int64_t *offset_recorder,
                                   int64_t axis){
  int64_t des_ndim = aitisa_tensor_ndim(*destination);
  int64_t *des_dims = aitisa_tensor_dims(*destination);
  int64_t src_size = aitisa_tensor_size(source);
  uint8_t ele_size = aitisa_tensor_data_type(source).size;
  int64_t linear_idx;
  char* src_data = aitisa_tensor_data(source);
  char* des_data = aitisa_tensor_data(*destination);
  for(int64_t i=0; i<src_size; i++){
    // get linear index of current element
    linear_idx = 0;
    for(int j=0; j<des_ndim; j++){
      linear_idx += index_recorder[j] * offset_recorder[j];
    }
    // transport data to output
    memcpy(des_data+linear_idx*ele_size, src_data+i*ele_size, ele_size);
    // update index_recorder
    for(int j=des_ndim-1; j>0; j--){
      if(j == axis) continue;
      index_recorder[j] += 1;
      /* judge whether the index is out of boundary */
      if(index_recorder[j] >= des_dims[j]){
        index_recorder[j] = 0;
      }else{
        break;
      }
    }
  }
}

static Status batch_norm_without_channel(const Tensor input,
                   const Tensor scale, const Tensor bias,
                   const Tensor mean, const Tensor denominator,
                   Tensor *output){
  int64_t *dims = aitisa_tensor_dims(input);
  int64_t batch_size = dims[0];
  int64_t ndim = 2;
  // make parameters for slice
  int *begin =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*begin)*ndim);
  int *size =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*size)*ndim);
  int *step =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*step)*ndim);
  if(!begin || !size || !step) return STATUS_ALLOC_FAILED;
  for(int64_t i=0; i<ndim; i++){
    begin[i] = 0;
    size[i] = dims[i];
    step[i] = 1;
  }
  size[0] = 1;
  // make parameters for squeeze
  int64_t num_axis = 1;
  int64_t *axis =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*axis)*num_axis);
  if(!axis) return STATUS_ALLOC_FAILED;
  axis[0] = 0;
  // implement 1d batch normalization without channel
  uint8_t ele_size = aitisa_tensor_data_type(input).size;
  char* out_data = aitisa_tensor_data(*output);
  for(int64_t s=0; s<batch_size; s++){
    Tensor temp0, temp1;
    CHECK_STATUS(
      aitisa_slice(input, begin, size, step, &temp0));
    CHECK_STATUS(
      aitisa_squeeze(temp0, axis, num_axis, &temp1));
    CHECK_STATUS(
      aitisa_destroy(&temp0));
    CHECK_STATUS(
      aitisa_sub(temp1, mean, &temp0));
    CHECK_STATUS(
      aitisa_destroy(&temp1));
    CHECK_STATUS(
      aitisa_div(temp0, denominator, &temp1));
    CHECK_STATUS(
      aitisa_destroy(&temp0));
    CHECK_STATUS(
      aitisa_mul(scale, temp1, &temp0));
    CHECK_STATUS(
      aitisa_destroy(&temp1));
    CHECK_STATUS(
      aitisa_add(temp0, bias, &temp1));
    CHECK_STATUS(
      aitisa_destroy(&temp0));
    // transport data to output
    int64_t size = aitisa_tensor_size(temp1);
    char* temp1_data = aitisa_tensor_data(temp1);
    for(int64_t ele=0; ele<size; ele++){
      memcpy(out_data+s*size*ele_size+ele*ele_size,
             temp1_data+ele*ele_size, ele_size);
    }
    CHECK_STATUS(
      aitisa_destroy(&temp1));
    // update begin
    begin[0]++;
  }
  // destroy parameters for slice and squeeze
  aitisa_default_cpu_allocator()->raw_dealloc(begin);
  aitisa_default_cpu_allocator()->raw_dealloc(size);
  aitisa_default_cpu_allocator()->raw_dealloc(step);
  aitisa_default_cpu_allocator()->raw_dealloc(axis);
  return STATUS_SUCCESS;
}

static Status batch_norm_with_channel(const Tensor input, const int axis,
           const Tensor* scale_array, const Tensor* bias_array,
           const Tensor* mean_array, const Tensor* denominator,
           Tensor *output){
  int64_t ndim = aitisa_tensor_ndim(input);
  if(axis!=1 && axis!=ndim-1) return STATUS_INVALID_ARGUMENT;
  int64_t *dims = aitisa_tensor_dims(input);
  int64_t batch_size = dims[0];
  int64_t channel_size = dims[axis];
  // make parameters for slice
  int *begin =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*begin)*ndim);
  int *size =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*size)*ndim);
  int *step =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*step)*ndim);
  if(!begin || !step || !size) return STATUS_ALLOC_FAILED;
  for(int64_t i=0; i<ndim; i++){
    begin[i] = 0;
    step[i] = 1;
    size[i] = (int)(dims[i]);
  }
  size[0] = 1;
  size[axis] = 1;
  // make parameters for squeeze
  int64_t num_saxis = 2;
  int64_t *squeeze_axis =
    aitisa_default_cpu_allocator()->
      raw_alloc(sizeof(*squeeze_axis)*num_saxis);
  squeeze_axis[0] = 0;
  squeeze_axis[1] = axis;
  // make parameters for data transportation
  int64_t *index_recorder =
    aitisa_default_cpu_allocator()->
      raw_alloc(sizeof(*index_recorder)*ndim);
  int64_t *offset_recorder =
    aitisa_default_cpu_allocator()->
      raw_alloc(sizeof(*offset_recorder)*ndim);
  if(!index_recorder || !offset_recorder){
    return STATUS_ALLOC_FAILED;
  }
  for(int64_t i=0; i<ndim; i++){
    index_recorder[i] = 0;
  }
  offset_recorder[ndim-1] = 1;
  for(int i=ndim-2; i>=0 ;i--){
    offset_recorder[i] = offset_recorder[i+1] * dims[i+1];
  }
  // implement batch norm 
  for(int64_t s=0; s<batch_size; s++){
    // set begin
    begin[0] = (int)s;
    //set index_recorder
    index_recorder[0] = s;
    for(int64_t c=0; c<channel_size; c++){
      Tensor temp0, temp1;
      // slice to get a channel
      begin[axis] = (int)c;// set begin
      CHECK_STATUS(
        aitisa_slice(input, begin, size, step, &temp0));
      CHECK_STATUS(
        aitisa_squeeze(temp0, squeeze_axis, num_saxis, &temp1));
      CHECK_STATUS(
        aitisa_destroy(&temp0));
      // implement the equation
      CHECK_STATUS(
        aitisa_sub(temp1, mean_array[c], &temp0));
      CHECK_STATUS(
        aitisa_destroy(&temp1));
      CHECK_STATUS(
        aitisa_div(temp0, denominator[c], &temp1));
      CHECK_STATUS(
        aitisa_destroy(&temp0));
      CHECK_STATUS(
        aitisa_mul(scale_array[c], temp1, &temp0));
      CHECK_STATUS(
        aitisa_destroy(&temp1));
      CHECK_STATUS(
        aitisa_add(temp0, bias_array[c], &temp1));
      // transport data of temp1 to output
      index_recorder[axis] = c;// set index_recorder
      batch_norm_transport(temp1, output, index_recorder,
                           offset_recorder, axis);
      // destroy temp0 and temp1
      CHECK_STATUS(
        aitisa_destroy(&temp0));
      CHECK_STATUS(
        aitisa_destroy(&temp1));
    }
  }
  // destroy parameters for slice and squeeze
  aitisa_default_cpu_allocator()->raw_dealloc(begin);
  aitisa_default_cpu_allocator()->raw_dealloc(size);
  aitisa_default_cpu_allocator()->raw_dealloc(step);
  aitisa_default_cpu_allocator()->raw_dealloc(squeeze_axis);
  return STATUS_SUCCESS;
}

static Status get_intermediate_tensors(const Tensor input, const int axis,
                                  const Tensor scale, Tensor **scale_array,
                                  const Tensor bias, Tensor **bias_array,
                                  const Tensor mean, Tensor **mean_array,
                                  const Tensor variance, const double epsilon,
                                  Tensor **denominator){
  int64_t in_ndim = aitisa_tensor_ndim(input);
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  LayoutType layout = aitisa_tensor_layout_type(input);
  if(in_ndim == 2){
    // batch norm 1d without channel
    *scale_array = NULL;
    *bias_array = NULL;
    *mean_array = NULL;
    // get epsilon tensor
    Tensor eps_tensor, temp;
    int64_t *eps_dims = aitisa_tensor_dims(variance);
    int64_t eps_ndim = aitisa_tensor_ndim(variance);
    CHECK_STATUS(
      aitisa_full(dtype, device, eps_dims, eps_ndim, epsilon, &eps_tensor));
    // get sqrt(variance+epsilon), denoted by denominator
    CHECK_STATUS(
      aitisa_add(variance, eps_tensor, &temp));
    CHECK_STATUS(
      aitisa_sqrt(temp, *denominator));
    // destroy epsilon and temp tensor
    aitisa_destroy(&eps_tensor);
    aitisa_destroy(&temp);
  }else if(in_ndim > 2){
    // the case with channel
    int64_t *in_dims = aitisa_tensor_dims(input);
    int64_t num_channels = aitisa_tensor_dim(input, axis);
    /* get dimensions of parameters including scale_array,
     mean_array, bias_array and denominator */
    int64_t param_ndim = in_ndim - 2;
    int64_t *param_dims =
      aitisa_default_cpu_allocator()->
        raw_alloc(sizeof(*param_dims)*param_ndim);
    if(!param_dims) return STATUS_ALLOC_FAILED;
    if(axis == 1){
      for(int64_t i=2; i<in_ndim; i++){
        param_dims[i-2] = in_dims[i];
      }
    }else if(axis == in_ndim-1){
      for(int64_t i=1; i<in_ndim-1; i++){
        param_dims[i-1] = in_dims[i];
      }
    }
    // get epsilon tensor
    Tensor eps_tensor;
    CHECK_STATUS(
      aitisa_full(dtype, device, param_dims, param_ndim, epsilon, &eps_tensor));
    // get mean_array, scale_array, bias_array, denominator
    int64_t tensor_size = size_of_dims(param_dims, param_ndim);
    uint8_t ele_size = dtype.size;
    *mean_array =
      aitisa_default_cpu_allocator()->
        raw_alloc(sizeof(**mean_array)*num_channels);
    if(!*mean_array) return STATUS_ALLOC_FAILED;
    *scale_array =
      aitisa_default_cpu_allocator()->
        raw_alloc(sizeof(**scale_array)*num_channels);
    if(!*scale_array) return STATUS_ALLOC_FAILED;
    *bias_array =
      aitisa_default_cpu_allocator()->
        raw_alloc(sizeof(**bias_array)*num_channels);
    if(!*bias_array) return STATUS_ALLOC_FAILED;
    *denominator = NULL;
    *denominator =
      aitisa_default_cpu_allocator()->
        raw_alloc(sizeof(**denominator)*num_channels);
    if(!*denominator) return STATUS_ALLOC_FAILED;
    Tensor *var_array =
      aitisa_default_cpu_allocator()->
        raw_alloc(sizeof(*var_array)*num_channels);
    if(!var_array) return STATUS_ALLOC_FAILED;
    for(uint32_t c=0; c<num_channels; c++){
      CHECK_STATUS(
        aitisa_create(dtype, device, layout, param_dims,
                      param_ndim, &((*mean_array)[c])));
      CHECK_STATUS(
        aitisa_create(dtype, device, layout, param_dims,
                      param_ndim, &((*scale_array)[c])));
      CHECK_STATUS(
        aitisa_create(dtype, device, layout, param_dims,
                      param_ndim, &((*bias_array)[c])));
      CHECK_STATUS(
        aitisa_create(dtype, device, layout, param_dims,
                      param_ndim, &(var_array[c])));
      CHECK_STATUS(
        aitisa_create(dtype, device, layout, param_dims,
                      param_ndim, &((*denominator)[c])));
      // repeatedly copy data to array
      char* mean_array_c_data = aitisa_tensor_data((*mean_array)[c]);
      char* mean_data = aitisa_tensor_data(mean);
      char* scale_array_c_data = aitisa_tensor_data((*scale_array)[c]);
      char* scale_data = aitisa_tensor_data(scale);
      char* bias_array_c_data = aitisa_tensor_data((*bias_array)[c]);
      char* bias_data = aitisa_tensor_data(bias);
      char* var_array_c_data = aitisa_tensor_data(var_array[c]);
      char* var_data = aitisa_tensor_data(variance);
      for(uint32_t i=0; i<tensor_size; i++){
        memcpy((void*)(mean_array_c_data+i*ele_size),
               (void*)(mean_data+c*ele_size), ele_size);
        memcpy((void*)(scale_array_c_data+i*ele_size),
               (void*)(scale_data+c*ele_size), ele_size);
        memcpy((void*)(bias_array_c_data+i*ele_size),
               (void*)(bias_data+c*ele_size), ele_size);
        memcpy((void*)(var_array_c_data+i*ele_size),
               (void*)(var_data+c*ele_size), ele_size);
      }
      // sqrt(var_array[c] + epsilon_tensor)
      CHECK_STATUS(
        aitisa_add(var_array[c], eps_tensor, &((*denominator)[c])));
      CHECK_STATUS(
        aitisa_sqrt((*denominator)[c], &((*denominator)[c])));
      aitisa_destroy(&(var_array[c]));
    }
    aitisa_destroy(&eps_tensor);
  }
  return STATUS_SUCCESS;
}

Status aitisa_batch_norm(const Tensor input, const int axis,
                         const Tensor scale, const Tensor bias,
                         const Tensor mean, const Tensor variance,
                         const double epsilon, Tensor *output){
  // check whether parameters are matched
  CHECK_STATUS(
    batch_norm_check_parameters(input, axis, scale, bias,
                              mean, variance, epsilon));
  // create output
  CHECK_STATUS(
    batch_norm_create_output(input, output));
  //implement batch normalization
  Status status;
  int64_t ndim = aitisa_tensor_ndim(input);
  Tensor* scale_array, * bias_array, * mean_array, * denominators;
  if(ndim == 2){
    // get denominator which is sqrt(variance+epsilon)
    Tensor denominator;
    Tensor *denominator_ptr = &denominator;
    CHECK_STATUS(
      get_intermediate_tensors(input, axis, scale, &scale_array,
                         bias, &bias_array, mean, &mean_array,
                         variance, epsilon, &denominator_ptr));
    status = batch_norm_without_channel(input, scale, bias,
                                mean, denominator, output);
  }else if(ndim>2 && ndim<6){
    CHECK_STATUS(
      get_intermediate_tensors(input, axis, scale, &scale_array,
        bias, &bias_array, mean, &mean_array,
        variance, epsilon, &denominators));
    status = batch_norm_with_channel(input, axis, scale_array,
                                     bias_array, mean_array,
                                     denominators, output);
  }else{
    status = STATUS_DIMENSIONS_MISMATCH;
  }
  return status;
}
