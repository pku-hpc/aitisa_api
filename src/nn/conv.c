#include "src/core/utils.h"
#include "src/core/allocator.h"
#include "src/basic/index_utils.h"
#include "src/nn/conv.h"
#include <stdio.h>

/*
 * Compute the output dims based on input dims, filter dims and other related
 * configurations such as stride, padding, and dialation.
 */
static void conv_output_dims(const int64_t *input_dims,
                             const int64_t *filter_dims, const int *stride,
                             const int *padding, const int *dilation,
                             int64_t ndim, int64_t *output_dims) {
  output_dims[0] = input_dims[0];
  output_dims[1] = filter_dims[0];
  for (int64_t d = 2; d < ndim; ++d) {
    int64_t kernel = dilation[d - 2] * (filter_dims[d] - 1) + 1;
    output_dims[d] = (input_dims[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
}

/* The implementation of convolution for the float data type */
static Status aitisa_conv_float(const Tensor input, const Tensor filter,
                                const int *stride, const int *padding,
                                const int *dilation, const int groups,
                                Tensor *output_ptr) {
  int64_t ndim = aitisa_tensor_ndim(input);
  int64_t *in_dims = aitisa_tensor_dims(input);
  int64_t *fil_dims = aitisa_tensor_dims(filter);
  int64_t *out_dims =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_dims) * ndim);
  int64_t *in_coords =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*in_coords) * ndim);
  int64_t *fil_coords =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*fil_coords) * ndim);
  int64_t *out_coords =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_coords) * ndim);
  int64_t *tmp_coords =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*tmp_coords) * ndim);
  int64_t *in_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*in_strides) * ndim);
  int64_t *fil_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*fil_strides) * ndim);
  int64_t *out_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_strides) * ndim);

  conv_output_dims(in_dims, fil_dims, stride, padding, dilation, ndim, out_dims);

  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  // LayoutType layout_type = aitisa_tensor_layout_type(input);
  CHECK_STATUS(
      aitisa_create(dtype, device, out_dims, ndim, NULL, 0, &new_tensor));
  *output_ptr = new_tensor;

  aitisa_get_all_strides(input, in_strides);
  aitisa_get_all_strides(filter, fil_strides);
  aitisa_get_all_strides(*output_ptr, out_strides);

  float *in_data = aitisa_tensor_data(input);
  float *fil_data = aitisa_tensor_data(filter);
  float *out_data = aitisa_tensor_data(*output_ptr);

  int64_t out_space_size = size_from_dim(2, out_dims, ndim);
  // int64_t in_space_size = size_from_dim(2, in_dims, ndim);
  int64_t fil_space_size = size_from_dim(2, fil_dims, ndim);

  int64_t nbatch = aitisa_tensor_dim(input, 0);
  int64_t in_chan = aitisa_tensor_dim(input, 1);
  int64_t out_chan = aitisa_tensor_dim(*output_ptr, 1);
  int64_t sdim = ndim - 2;

  for (int64_t b = 0; b < nbatch; ++b) {
    for (int64_t oc = 0; oc < out_chan; ++oc) {
      for (int64_t out_idx = 0; out_idx < out_space_size; ++out_idx) {
        aitisa_linidx_to_coords(*output_ptr, out_idx, out_coords);
        out_coords[0] = b;
        out_coords[1] = oc;

        for (int64_t d = 0; d < sdim; ++d) {
          in_coords[2 + d] = out_coords[2 + d] * stride[d] - padding[d];
        }
        in_coords[0] = b;

        float tmp = 0.0;
        for (int64_t ic = 0; ic < in_chan; ++ic) {
          in_coords[1] = ic;
          for (int64_t fil_idx = 0; fil_idx < fil_space_size; ++fil_idx) {
            aitisa_linidx_to_coords(filter, fil_idx, fil_coords);
            fil_coords[0] = oc;
            fil_coords[1] = ic;

            int inside = 1;
            for (int64_t d = 0; d < sdim && inside; ++d) {
              tmp_coords[2 + d] =
                  in_coords[2 + d] + dilation[d] * fil_coords[2 + d];
              if (tmp_coords[2 + d] < 0 || tmp_coords[2 + d] >= in_dims[2 + d])
                inside = 0;
            }
            tmp_coords[0] = in_coords[0];
            tmp_coords[1] = in_coords[1];

            if (inside) {
              int64_t in_offset =
                  aitisa_coords_to_offset(input, tmp_coords, ndim);
              int64_t fil_offset =
                  aitisa_coords_to_offset(filter, fil_coords, ndim);
              tmp += in_data[in_offset] * fil_data[fil_offset];
            }
          }
          int64_t out_offset =
              aitisa_coords_to_offset(*output_ptr, out_coords, ndim);
          out_data[out_offset] = tmp;
        }
      }
    }
  }

  aitisa_default_cpu_allocator()->raw_dealloc(out_dims);
  aitisa_default_cpu_allocator()->raw_dealloc(in_coords);
  aitisa_default_cpu_allocator()->raw_dealloc(fil_coords);
  aitisa_default_cpu_allocator()->raw_dealloc(out_coords);
  aitisa_default_cpu_allocator()->raw_dealloc(in_strides);
  aitisa_default_cpu_allocator()->raw_dealloc(fil_strides);
  aitisa_default_cpu_allocator()->raw_dealloc(out_strides);
  aitisa_default_cpu_allocator()->raw_dealloc(tmp_coords);
}

/*
 * The implementation of the convolution operation. For now, it is only support
 * the float data type.
 */
Status aitisa_conv(const Tensor input, const Tensor filter, const int *stride,
                   const int *padding, const int *dilation, const int groups,
                   Tensor *output_ptr) {
  DataType dtype = aitisa_tensor_data_type(input);
  switch (dtype.code) {
    case TYPE_FLOAT:
      CHECK_STATUS(aitisa_conv_float(input, filter, stride, padding, dilation,
                                     groups, output_ptr));
      break;
    default:
      return STATUS_NOT_SUPPORTED;
  }
}

Status aitisa_conv2d(const Tensor input, const Tensor filter, const int *stride,
                     const int stride_len, const int *padding, const int padding_len, 
                     const int *dilation, const int dilation_len, const int groups,
                     Tensor *output_ptr) {
  int64_t ndim = aitisa_tensor_ndim(input);
  if(ndim != 4){
    return STATUS_DIMENSIONS_MISMATCH;
  }

  return aitisa_conv(input, filter, stride, padding, dilation, groups, output_ptr);
}

Status aitisa_conv3d(const Tensor input, const Tensor filter, const int *stride,
                     const int stride_len, const int *padding, const int padding_len, 
                     const int *dilation, const int dilation_len, const int groups,
                     Tensor *output_ptr) {
  int64_t ndim = aitisa_tensor_ndim(input);
  if(ndim != 5){
    return STATUS_DIMENSIONS_MISMATCH;
  }

  return aitisa_conv(input, filter, stride, padding, dilation, groups, output_ptr);
}

