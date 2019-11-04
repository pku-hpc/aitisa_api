#ifndef SQUEEZE_H
#define SQUEEZE_H

#include "src/core/tensor.h"

/**
 * @brief Applies a squeeze over an input tensor along specified axises,
          deleting the dimensions of 1.
 * @param input The input tensor.
 * @param axis The axis array to point out which dimension to delete.
 * @param num_axis The length of array axis.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_squeeze(const Tensor input, int64_t *axis,
                                        int64_t num_axis, Tensor *output);


#endif // SQUEEZE_H
