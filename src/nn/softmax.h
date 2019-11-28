#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "src/core/tensor.h"

/**
 * @brief Applies a softmax over an input signal along a specified axis.
 *
 * @param input The input tensor.
 * @param axis The axis along which a softmax over input is applied.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_softmax(const Tensor input, const int axis,
										Tensor *output);
#endif
