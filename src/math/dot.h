#ifndef DOT_H
#define DOT_H

#include "src/core/tensor.h"

/**
 * @brief Applies a dot over two input signals.
 * @details If the two inputs are one-dimension, then implement the normal dot;
 * 			If the two inputs are two-dimension, then implement the matmul;
 * 			If one of the two inputs is a scalar, then implement the scalar-tensor multiplication;
 * 			If the second input is one-dimension, then calculate the dot between the last dimension 
 * 			of the first input and the second input;
 * 			If both of the dimensions of the two inputs are larger than 2, then calculate the dot 
 * 			between the last dimension of the first input and the second-to-last dimension of the 
 * 			second input: dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m]).
 *
 * @param tensor1 The first input tensor.
 * @param tensor2 The second input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */

AITISA_API_PUBLIC Status aitisa_dot(const Tensor tensor1, const Tensor tensor2,
                  Tensor *output);
#endif // DOT_H
