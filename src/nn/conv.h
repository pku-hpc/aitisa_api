#ifndef CONV_H
#define CONV_H

#include "src/core/tensor.h"

/**
 * @brief Applies a convolution over an input image with a filter.
 *
 * @param input The input tensor of a shape [num_batches, in_channels,
 *              in_spatial_shapes].
 * @param filter The filter tensor of a shape [out_channnels, in_channels,
 *               in_spatial_shapes].
 * @param stride The stride of a shape [spatial_shapes] each of which is >= 1.
 * @param padding The padding of a shape [spatial_shapes] each of which is >=
 *                0. The left and right of each dimension use the same padding *
 *                width.
 * @param dilation The dilation of a shape [spatial_shapes] each of which is
 *                 >= 1. This is a regular convolution if all values are 1.
 * @param groups An integer used to split input into groups, in_channels should
 *               be divisible by the number of groups.
 * @param output The output tensor pointer of a shape [num_batches,
 *               out_channels, * * out_spatial_shapes].
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_conv(const Tensor input, const Tensor filter,
                                     const int *stride, const int *padding,
                                     const int *dilation, const int groups,
                                     Tensor *output);

AITISA_API_PUBLIC Status aitisa_conv2d(const Tensor input, const Tensor filter, 
                                       const int *stride, const int stride_len, 
                                       const int *padding, const int padding_len, 
                                       const int *dilation, const int dilation_len, 
                                       const int groups, Tensor *output_ptr);

AITISA_API_PUBLIC Status aitisa_conv3d(const Tensor input, const Tensor filter, 
                                       const int *stride, const int stride_len, 
                                       const int *padding, const int padding_len, 
                                       const int *dilation, const int dilation_len, 
                                       const int groups, Tensor *output_ptr);
#endif
