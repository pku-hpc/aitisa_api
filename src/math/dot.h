#ifndef DOT_H
#define DOT_H

#include "src/core/tensor.h"

/**
 * @brief Applies a dot over two input signals.
 * @param tensor1 The first input tensor.
 * @param tensor2 The second input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_dot(const Tensor tensor1, const Tensor tensor2,
                  Tensor *output);
#endif // DOT_H
