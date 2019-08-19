#ifndef MATMUL_H
#define MATMUL_H

#include "src/core/tensor.h"

AITISA_API_PUBLIC Status aitisa_matmul(const Tensor tensor1, const Tensor tensor2, Tensor *output);

#endif