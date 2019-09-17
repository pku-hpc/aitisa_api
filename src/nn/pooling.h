#ifndef POOLING_H
#define POOLING_H
Status aitisa_pooling(const Tensor input, const char *mode,
                      const int *ksize, const int *stride,
                      const int *padding, Tensor *output);
#endif // POOLING_H
