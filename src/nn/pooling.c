#include "src/core/allocator.h"
#include "src/core/dispatch.h"
#include "src/basic/index_utils.h"
#include "src/nn/pooling.h"
#include <math.h>

#define pooling_kernel(typename)                      \
  if(!strcmp(mode, "avg")){                           \
    temp += ((typename*)in_data)[linear_idx];         \
  }                                                   \
  else if (!strcmp(mode, "max")) {                    \
    if (temp < ((typename*)in_data)[linear_idx]) {    \
      temp = ((typename*)in_data)[linear_idx];        \
    }                                                 \
  }

#define pooling_output_element(typename)                                      \
  if(!strcmp(mode, "avg")){                                                   \
    ((typename*)out_data)[out_linear_idx] = (typename)(temp / window_size);   \
  }                                                                           \
  else if (!strcmp(mode, "max")) {                                            \
    ((typename*)out_data)[out_linear_idx] = (typename)temp;                   \
  }


static Status pooling_create_output(const Tensor input, int64_t *out_dims,
                                    const int *ksize,	  const int *stride,
                                    const int *padding, const int *dilation,
                                    const int axis,     int64_t c_start,
                                    Tensor *output){
  int64_t ndim = aitisa_tensor_ndim(input);
  int64_t *in_dims = aitisa_tensor_dims(input);
  out_dims[0] = in_dims[0];
  out_dims[axis] = in_dims[axis];
  for(int64_t i = c_start; i < c_start + ndim - 2; i++){
    double temp = in_dims[i] + 2.0 * padding[i - c_start] -
                  dilation[i - c_start] * (ksize[i - c_start] - 1);
    out_dims[i] = (int64_t)floor((temp-1)/stride[i-c_start] + 1 );
  }
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  // LayoutType layout = aitisa_tensor_layout_type(input);
  return
    aitisa_create(dtype, device, out_dims, ndim, NULL, 0, output);
}

static Status pooling_double(const Tensor input, const char *mode,
                             const int *ksize,	 const int *stride,
                             const int *padding, const int *dilation,
                             const int axis,     Tensor *output){
  int64_t ndim = aitisa_tensor_ndim(input);
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t c_start = axis==1?2:1;
  int64_t *out_dims =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_dims)*ndim);
  int64_t *index =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*index)*ndim);
  int64_t *strides =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*strides)*ndim);
  int64_t *out_strides =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_strides)*ndim);
  int64_t *k_head =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*k_head)*ndim);
  int64_t *k_tail =
    aitisa_default_cpu_allocator()->raw_alloc(sizeof(*k_tail)*ndim);
  // Check whether the kernel window is larger than a channel
  for(int64_t i=0; i<ndim-2; i++){
    if((ksize[i]-1)*dilation[i]+1 > 2*padding[i]+in_dims[i+c_start]){
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // Create output
  Tensor new_tensor;
  CHECK_STATUS(
    pooling_create_output(input, out_dims, ksize, stride,
                   padding, dilation, axis, c_start, &new_tensor));
  // Implement pooling
  int64_t window_size = 1;
  for(int64_t i=0; i<ndim-2; i++){
    window_size *= ksize[i];
  }
  aitisa_get_all_strides(input, strides);
  aitisa_get_all_strides(new_tensor, out_strides);
  void *in_data = aitisa_tensor_data(input);
  void *out_data = aitisa_tensor_data(new_tensor);
  DataType dtype = aitisa_tensor_data_type(input);
  int64_t batch_size = in_dims[0];
  int64_t num_channel = in_dims[axis];
  for(int64_t s=0; s<batch_size; s++){
    index[0] = s;
    k_tail[0] = s;
    k_head[0] = s;
    for(int64_t c=0; c<num_channel; c++){
      index[axis] = c;
      k_tail[axis] = c;
      k_head[axis] = c;
      int64_t bc_stride = s * strides[0] + c * strides[axis];
      int64_t out_bc_stride = s * out_strides[0] + c * out_strides[axis];
      // Set k_head and k_tail according to padding and dilation
      for(int64_t i=c_start; i<c_start+ndim-2; i++){
        k_head[i] = 0 - padding[i-c_start];
        k_tail[i] = k_head[i]+(ksize[i-c_start]-1)*dilation[i-c_start];
      }
      int window_idx = 0;
      int hasWindow = 1;
      while(hasWindow){
        // set index
        for(int64_t i=c_start; i<c_start+ndim-2; i++){
          index[i] = k_head[i];
        }
        double temp = 0; 
        int hasElement = 1;
        while(hasElement){
          int64_t linear_idx = bc_stride;
          int padded = 0;
          for(int64_t i=c_start; i<c_start+ndim-2; i++){
            if(0<=index[i] && index[i]<in_dims[i]){
              linear_idx += index[i] * strides[i];
            }else{
              padded = 1;
              break;
            }
          }
          // need to consider padding
          if(!padded){
            AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, pooling_kernel);
          }
          // update index
          for(int64_t i=c_start+ndim-3; i>=c_start; i--){
            index[i] += dilation[i-c_start];
            if(index[i] > k_tail[i]){
              // judge if there is still element
              if(i == c_start) hasElement = 0;
              index[i] = k_head[i];
            }else{
              break;
            }
          }
        }
        // update new_tensor
        int64_t out_linear_idx =
          window_idx * out_strides[c_start+ndim-3] + out_bc_stride;
        AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, pooling_output_element);
        // update k_head and k_tail
        for(int64_t i=c_start+ndim-3; i>=c_start; i--){
          k_head[i] = k_head[i] + stride[i-c_start];
          k_tail[i] = k_head[i]+(ksize[i-c_start]-1)*dilation[i-c_start];
          if(k_tail[i] > padding[i-c_start]+in_dims[i]-1){
            // judge if there is still window
            if(i == c_start) hasWindow = 0;
            k_head[i] = - padding[i-c_start];
            k_tail[i] = k_head[i]+(ksize[i-c_start]-1)*dilation[i-c_start];
          }else{
            break;
          }
        }
        window_idx++;
      }
    }
  }
  *output = new_tensor;
  // Destroy temporary variables
  aitisa_default_cpu_allocator()->raw_dealloc(out_dims);
  aitisa_default_cpu_allocator()->raw_dealloc(index);
  aitisa_default_cpu_allocator()->raw_dealloc(k_head);
  aitisa_default_cpu_allocator()->raw_dealloc(k_tail);
  aitisa_default_cpu_allocator()->raw_dealloc(strides);
  aitisa_default_cpu_allocator()->raw_dealloc(out_strides);

  return STATUS_SUCCESS;
}

Status aitisa_pooling(const Tensor input, const char *mode,
                      const int *ksize,	  const int *stride,
                      const int *padding, const int *dilation,
                      Tensor *output){
    Status status;
    int64_t ndim = aitisa_tensor_ndim(input);
    if(ndim>2 && ndim<6){
      int axis = 1;
      status = pooling_double(input, mode, ksize, stride,
                                padding, dilation, axis, output);
    }else{
      status = STATUS_DIMENSIONS_MISMATCH;
    }
    return status;
}
