#ifndef BINARYOP_H
#define BINARYOP_H

#include "src/core/tensor.h"

typedef enum {
  OP_ADD = 0U,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_NOPS = 4U
} OpCode;

typedef void (*BinaryOpFunc)(void *a, void *b, OpCode op, void *c);
BinaryOpFunc aitisa_binary_op_func(DataType dtype);

AITISA_API_PUBLIC Status aitisa_add(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);
AITISA_API_PUBLIC Status aitisa_sub(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);
AITISA_API_PUBLIC Status aitisa_mul(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);
AITISA_API_PUBLIC Status aitisa_div(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

#endif