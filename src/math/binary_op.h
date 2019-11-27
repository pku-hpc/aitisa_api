#ifndef BINARYOP_H
#define BINARYOP_H

#include "src/core/tensor.h"

/**
 * @brief Enumeration type for all possible binary operation types
 *
 * @details Use to select the right operator in calculation
 */
typedef enum {
  OP_ADD = 0U,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_NOPS = 4U /**< The total number of all possible operations */
} OpCode;

typedef void (*BinaryOpFunc)(void *a, void *b, OpCode op, void *c);
BinaryOpFunc aitisa_binary_op_func(DataType dtype);

/**
 * @brief Add two tensors in element-wise way
 *
 * @param tensor1 One of the input tensors to be added
 * @param tensor2 One of the input tensors to be added
 * @param output Output tensor contains the result
 *
 * @return
 * @retval STATUS_SUCCESS Successfully add two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_add(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);
/**
 * @brief Subtract two tensors in element-wise way
 *
 * @param tensor1 One of the input tensors to be subtracted
 * @param tensor2 One of the input tensors to be subtracted
 * @param output Output tensor contains the result
 *
 * @return
 * @retval STATUS_SUCCESS Successfully subtract two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_sub(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

/**
 * @brief Multiple two tensors in element-wise way
 *
 * @param tensor1 One of the input tensors to be Multipled
 * @param tensor2 One of the input tensors to be Multipled
 * @param output Output tensor contains the result
 *
 * @return
 * @retval STATUS_SUCCESS Successfully Multiple two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_mul(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

/**
 * @brief Divide two tensors in element-wise way
 *
 * @param tensor1 One of the input tensors to be Divided
 * @param tensor2 One of the input tensors to be Divided
 * @param output Output tensor contains the result
 *
 * @return
 * @retval STATUS_SUCCESS Successfully Multiple two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_div(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

#endif
