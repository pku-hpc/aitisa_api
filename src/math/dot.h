#ifndef DOT_H
#define DOT_H
#include "src/core/tensor.h"
Status aitisa_dot(const Tensor tensor1, const Tensor tensor2,
                  Tensor *output);
#endif // DOT_H
