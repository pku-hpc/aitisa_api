#ifndef POOLING_H
#define POOLING_H

#include "src/core/tensor.h"

/**
 * @brief Applies a pooling over an input signal composed of several feature maps.
 *
 * @param input The input tensor of a shape [num_batches, in_channels,
 *              in_spatial_shapes].
 * @param mode The pooling mode including "max" and "avg".
 * @param ksize The window size of a shape each of which is >= 1.
 * @param stride The stride of a shape [spatial_shapes] each of which is >= 1.
 * @param padding The padding of a shape [spatial_shapes] each of which is >=
 *                0. If the padding number of some axis is non-zero, implicit 
 *                zero padding is added to input on both sides. If the padding 
 *                number of some axis is odd, then the rest zeros will be placed
 *                on the high-dimension side.
 * @param dilation The dilation of a shape [spatial_shapes] each of which is
 *                 >= 1. This is a regular pooling if all values are 1.
 * @param output The output tensor pointer of a shape [num_batches,
 *               out_channels, * * out_spatial_shapes].
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_pooling(const Tensor input, const char *mode,
										const int *ksize,   const int *stride,
										const int *padding, const int *dilation,
										Tensor *output);
#endif
