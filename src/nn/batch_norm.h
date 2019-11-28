#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "src/core/tensor.h"

/**
 * @brief Applies a batch normalization over an input along axis which is
 *        actually the axis of channel. The data type of input, scale, bias, 
 *        mean and variance should be identical. The shape of output is same 
 *        as the shape of input.
 * 
 * @param input The input tensor, whose shape should be [N,L], [N,C,L],
 *              [N,L,C], [N,C,H,W], [N,H,W,C], [N,C,D,H,W] or [N,D,H,W,C].
 * @param axis The axis along which to be normalized, actually the axis of
 *             channel. For instance, if the shape of given input is [N,C,H,W], 
 *             then axis should be set to 1. But if the shape of input is [N,L], 
 *             axis should be 1.
 * @param scale Gamma in equations. If the shape of input is [N,L], then the
 *              shape of scale is [L]; else the shape of scale is [C].
 * @param bias Beta in equations. If the shape of input is [N,L], then the
 *             shape of bias is [L]; else the shape of bias is [C].
 * @param mean The mean tensor. If the shape of input is [N,L], then the
 *             shape of mean is [L]; else the shape of mean is [C].
 * @param variance The variance tensor. If the shape of input is [N,L], then the 
 *                 shape of variance is [L]; else the shape of variance is [C].
 * @param epsilon Small number added to variance to avoid dividing by zero.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_batch_norm(const Tensor input, const int axis,
                                           const Tensor scale, const Tensor bias,
                                           const Tensor mean, const Tensor variance,
                                           const double epsilon, Tensor *output);

#endif // BATCH_NORM_H
