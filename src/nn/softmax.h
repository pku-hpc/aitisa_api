#ifndef POOLING_H
#define POOLING_H

#include "src/core/tensor.h"

/**
 * @brief Applies a softmax over an input signal along a specified axis.
 * @param input The input tensor.
 * @param axis The axis along which a softmax over input is applied. 
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */

// FIXME: The special case of axis(axis = -1) should be explained in the brief.
AITISA_API_PUBLIC Status aitisa_softmax(const Tensor input, const int axis,
										                    Tensor *output);
#endif
