#ifndef CONV_H
#define CONV_H
#include "src/core/tensor.h"

AITISA_API_PUBLIC Status aitisa_conv(const Tensor input, const Tensor filter,
                                     const int *stride, const int *padding,
                                     const int *dilation, const int groups,
                                     Tensor *output);
#endif
