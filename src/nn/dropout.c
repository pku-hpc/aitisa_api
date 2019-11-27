#include "src/nn/dropout.h"
#include <time.h>
#include "src/basic/duplicate.h"

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
  // implement dropout kernel
  DataType dtype = aitisa_tensor_data_type(*tensor);
  Status status = STATUS_SUCCESS;
  srand(time(NULL));
  int rate_line = (int)(rate * 100);
  switch (dtype.code) {
    case TYPE_INT8: {
      dropout_kernel(int8_t);
      break;
    }
    case TYPE_UINT8: {
      dropout_kernel(uint8_t);
      break;
    }
    case TYPE_INT16: {
      dropout_kernel(int16_t);
      break;
    }
    case TYPE_UINT16: {
      dropout_kernel(uint16_t);
      break;
    }
    case TYPE_INT32: {
      dropout_kernel(int32_t);
      break;
    }
    case TYPE_UINT32: {
      dropout_kernel(uint32_t);
      break;
    }
    case TYPE_INT64: {
      dropout_kernel(int64_t);
      break;
    }
    case TYPE_UINT64: {
      dropout_kernel(uint64_t);
      break;
    }
    case TYPE_FLOAT: {
      dropout_kernel(float);
      break;
    }
    case TYPE_DOUBLE: {
      dropout_kernel(double);
      break;
    }
    default:
      status = STATUS_NOT_SUPPORTED;
  }
  return status;
}

Status aitisa_dropout(const Tensor input, const double rate, Tensor *output) {
  // check if rate satisfy 0 <= rate <= 1
  if (rate < 0 || rate > 1) {
    return STATUS_INVALID_ARGUMENT;
  }
  // copy input
  Tensor new_tensor;
  CHECK_STATUS(aitisa_duplicate(input, &new_tensor));
  // implement dropout
  CHECK_STATUS(dropout_template(&new_tensor, rate));
  *output = new_tensor;
  return STATUS_SUCCESS;
}
