#ifndef DROPOUT_H
#define DROPOUT_H

#include "src/core/tensor.h"

/**
 * @brief Applies a dropout over an input signal with a rate specifying
 *        probability of an element to be zeroed.
 *
 * @param input The input tensor.
 * @param rate probability of an element to be zeroed.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_dropout(const Tensor input, const double rate,
                                        Tensor *output);

#endif // DROPOUT_H
