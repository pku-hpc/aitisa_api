#include "src/nn/dropout.h"
#include <time.h>
#include "src/basic/duplicate.h"
#include "src/core/dispatch.h"

#define dropout_kernel(typename)                            \
  typename *data = (typename *)aitisa_tensor_data(*tensor); \
  for (int64_t i = 0; i < size; i++) {                      \
    int random_num = rand() % 100 + 1;                      \
    if (random_num <= rate_line) {                          \
      data[i] = 0;                                          \
    }                                                       \
  }

static Status dropout_template(Tensor *tensor, const double rate) {
  int64_t size = aitisa_tensor_size(*tensor);
  // Implement dropout kernel
  DataType dtype = aitisa_tensor_data_type(*tensor);
  Status status = STATUS_SUCCESS;
  srand(time(NULL));
  int rate_line = (int)(rate * 100);
  AITISA_DISPATCH_ALL_TYPES_RETURN(dtype, dropout_kernel);
  return status;
}

Status aitisa_dropout(const Tensor input, const double rate, Tensor *output) {
  // Check if rate satisfy 0 <= rate <= 1
  if (rate < 0 || rate > 1) {
    return STATUS_INVALID_ARGUMENT;
  }
  // Copy input
  Tensor new_tensor;
  CHECK_STATUS(aitisa_duplicate(input, &new_tensor));
  // Implement dropout
  CHECK_STATUS(dropout_template(&new_tensor, rate));
  *output = new_tensor;
  return STATUS_SUCCESS;
}
