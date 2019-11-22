#ifndef MATMUL_H
#define MATMUL_H

#include "src/core/tensor.h"

/**
 * @brief Matrix multiplication of tensor1 and tensor2.
 *
 * @details Apply different kernel(dot, gemv, gemm, batch_gemm) according
 * the dim size of inputs:
 * (0) ndim_tensor1 <  1 & ndim_tensor2 < 1  ==> INVALID ERROR.
 * (1) ndim_tensor1 == 1 & ndim_tensor2 == 1 ==> output = dot(tensor1, tensor2);
 * (2) ndim_tensor1 == 1 & ndim_tensor2 == 2 ==> tensor1 is considered as a
 * one-row matrix, output = gemm(tensor1, tensor2);
 * (3) ndim_tensor1 == 1 & ndim_tensor2 >= 3 ==> tensor1 is considered
 * as a one-row matrix, output = batch_gemm(tensor1, tensor2);
 * (4) ndim_tensor1 == 2 & ndim_tensor2 == 1 ==> output = gemv(tensor1, tensor2);
 * (5) ndim_tensor1 == 2 & ndim_tensor2 == 2 ==> output = gemm(tensor1, tensor2);
 * (6) ndim_tensor1 == 2 & ndim_tensor2 >= 3 ==> output = batch_gemm(tensor1, tensor2);
 * (7) ndim_tensor1 >= 3 & ndim_tensor2 == 1 ==> tensor2 is considered as a
 * one-column matrix, output = batch_gemm(tensor1, tensor2);
 * (8) ndim_tensor1 >= 3 & ndim_tensor2 >= 2 ==> output = batch_gemm(tensor1, tensor2);
 *
 * @param tensor1 The first input.
 * @param tensor2 The second input.
 * @param output Pointer to the result tensor, the output will be resized inside the function.
 *
 * @return Status.
 * @retval STATUS_SUCCESS Success.
 * @retval STATUS_NOT_SUPPORTED The device type or data type is not supported.
 * @retval STATUS_INVALID_ARGUMENT The dims of inputs are invalid.
 * @retval STATUS_DIMENSIONS_MISMATCH The dims of inputs are mismatched.
 *
 * @note (1) Only CPU is supported for now. (2) before doing batch_gemm, the batch dims
 * of tensor1 and tensor2 will be broadcasted.
 */
AITISA_API_PUBLIC Status aitisa_matmul(const Tensor tensor1,
                                       const Tensor tensor2, Tensor *output);

#endif
