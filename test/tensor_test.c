#include "../src/core/tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Tensor output;
  DataType dtype = {kInt32, sizeof(int)};
  Layout layout = {kDense};
  Device device = {kCPU, 0};
  int64_t dims[4] = {3, 3, 3, 3};
  // aitisa_create(dtype, layout, device, dims, 4, &output);
  int value  = 5;
  aitisa_full(dtype, device, dims, 4, (void *)&value, &output);
  printf("%ld\n", output.size);
  int *data = (int *)&output.storage->data;
  for (int i = 0; i < output.size; i++) {
    printf("%d ", data[i]);
  }
}