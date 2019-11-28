#ifndef SIGMOID_H
#define SIGMOID_H

#include "src/core/tensor.h"

/**
 * @brief Applies a sigmoid over an input signal.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_sigmoid(const Tensor input, Tensor *output);
#endif
