#ifndef SQRT_H
#define SQRT_H

#include "src/core/tensor.h"

/**
 * @brief Applies a square root over an input tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_sqrt(const Tensor input, Tensor *output);

#endif // SQRT_H
