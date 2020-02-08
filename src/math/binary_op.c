#include "src/math/binary_op.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"

void calc_int8_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(int8_t *)c = (*(int8_t *)a) + (*(int8_t *)b);
      break;
    case OP_SUB:
      *(int8_t *)c = (*(int8_t *)a) - (*(int8_t *)b);
      break;
    case OP_MUL:
      *(int8_t *)c = (*(int8_t *)a) * (*(int8_t *)b);
      break;
    case OP_DIV:
      *(int8_t *)c = (*(int8_t *)a) / (*(int8_t *)b);
      break;
    default:
      break;
  }
}

void calc_uint8_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(uint8_t *)c = (*(uint8_t *)a) + (*(uint8_t *)b);
      break;
    case OP_SUB:
      *(uint8_t *)c = (*(uint8_t *)a) - (*(uint8_t *)b);
      break;
    case OP_MUL:
      *(uint8_t *)c = (*(uint8_t *)a) * (*(uint8_t *)b);
      break;
    case OP_DIV:
      *(uint8_t *)c = (*(uint8_t *)a) / (*(uint8_t *)b);
      break;
    default:
      break;
  }
}

void calc_int16_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(int16_t *)c = (*(int16_t *)a) + (*(int16_t *)b);
      break;
    case OP_SUB:
      *(int16_t *)c = (*(int16_t *)a) - (*(int16_t *)b);
      break;
    case OP_MUL:
      *(int16_t *)c = (*(int16_t *)a) * (*(int16_t *)b);
      break;
    case OP_DIV:
      *(int16_t *)c = (*(int16_t *)a) / (*(int16_t *)b);
      break;
    default:
      break;
  }
}

void calc_uint16_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(uint16_t *)c = (*(uint16_t *)a) + (*(uint16_t *)b);
      break;
    case OP_SUB:
      *(uint16_t *)c = (*(uint16_t *)a) - (*(uint16_t *)b);
      break;
    case OP_MUL:
      *(uint16_t *)c = (*(uint16_t *)a) * (*(uint16_t *)b);
      break;
    case OP_DIV:
      *(uint16_t *)c = (*(uint16_t *)a) / (*(uint16_t *)b);
      break;
    default:
      break;
  }
}

void calc_int32_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(int32_t *)c = (*(int32_t *)a) + (*(int32_t *)b);
      break;
    case OP_SUB:
      *(int32_t *)c = (*(int32_t *)a) - (*(int32_t *)b);
      break;
    case OP_MUL:
      *(int32_t *)c = (*(int32_t *)a) * (*(int32_t *)b);
      break;
    case OP_DIV:
      *(int32_t *)c = (*(int32_t *)a) / (*(int32_t *)b);
      break;
    default:
      break;
  }
}

void calc_uint32_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(uint32_t *)c = (*(uint32_t *)a) + (*(uint32_t *)b);
      break;
    case OP_SUB:
      *(uint32_t *)c = (*(uint32_t *)a) - (*(uint32_t *)b);
      break;
    case OP_MUL:
      *(uint32_t *)c = (*(uint32_t *)a) * (*(uint32_t *)b);
      break;
    case OP_DIV:
      *(uint32_t *)c = (*(uint32_t *)a) / (*(uint32_t *)b);
      break;
    default:
      break;
  }
}

void calc_int64_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(int64_t *)c = (*(int64_t *)a) + (*(int64_t *)b);
      break;
    case OP_SUB:
      *(int64_t *)c = (*(int64_t *)a) - (*(int64_t *)b);
      break;
    case OP_MUL:
      *(int64_t *)c = (*(int64_t *)a) * (*(int64_t *)b);
      break;
    case OP_DIV:
      *(int64_t *)c = (*(int64_t *)a) / (*(int64_t *)b);
      break;
    default:
      break;
  }
}

void calc_uint64_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(uint64_t *)c = (*(uint64_t *)a) + (*(uint64_t *)b);
      break;
    case OP_SUB:
      *(uint64_t *)c = (*(uint64_t *)a) - (*(uint64_t *)b);
      break;
    case OP_MUL:
      *(uint64_t *)c = (*(uint64_t *)a) * (*(uint64_t *)b);
      break;
    case OP_DIV:
      *(uint64_t *)c = (*(uint64_t *)a) / (*(uint64_t *)b);
      break;
    default:
      break;
  }
}

void calc_float_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(float *)c = (*(float *)a) + (*(float *)b);
      break;
    case OP_SUB:
      *(float *)c = (*(float *)a) - (*(float *)b);
      break;
    case OP_MUL:
      *(float *)c = (*(float *)a) * (*(float *)b);
      break;
    case OP_DIV:
      *(float *)c = (*(float *)a) / (*(float *)b);
      break;
    default:
      break;
  }
}

void calc_double_value(void *a, void *b, OpCode op, void *c) {
  switch (op) {
    case OP_ADD:
      *(double *)c = (*(double *)a) + (*(double *)b);
      break;
    case OP_SUB:
      *(double *)c = (*(double *)a) - (*(double *)b);
      break;
    case OP_MUL:
      *(double *)c = (*(double *)a) * (*(double *)b);
      break;
    case OP_DIV:
      *(double *)c = (*(double *)a) / (*(double *)b);
      break;
    default:
      break;
  }
}

BinaryOpFunc binary_op_func[TYPE_NTYPES] = {
    calc_int8_value,  calc_uint8_value,  calc_int16_value, calc_uint16_value,
    calc_int32_value, calc_uint32_value, calc_int64_value, calc_uint64_value,
    calc_float_value, calc_double_value};

BinaryOpFunc aitisa_binary_op_func(DataType dtype) {
  return binary_op_func[dtype.code];
}

Status aitisa_add(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_binary_op_func(dtype)(a, b, OP_ADD, c);
    aitisa_set_typed_array_value_func(dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_sub(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_binary_op_func(dtype)(a, b, OP_SUB, c);
    aitisa_set_typed_array_value_func(dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_mul(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_binary_op_func(dtype)(a, b, OP_MUL, c);
    aitisa_set_typed_array_value_func(dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}

Status aitisa_div(const Tensor tensor1, const Tensor tensor2, Tensor *output) {
  if (aitisa_tensor_device(tensor1).type != DEVICE_CPU ||
      aitisa_tensor_device(tensor2).type != DEVICE_CPU) {
    return STATUS_NOT_SUPPORTED;
  }
  if (aitisa_tensor_data_type(tensor1).code !=
      aitisa_tensor_data_type(tensor2).code) {
    return STATUS_TYPE_MISMATCH;
  }
  int64_t ndim_tensor1 = aitisa_tensor_ndim(tensor1);
  int64_t ndim_tensor2 = aitisa_tensor_ndim(tensor2);
  // The dimension of two operators should be consistent, broadcast is not
  // support yet
  if (ndim_tensor1 != ndim_tensor2) {
    return STATUS_INVALID_ARGUMENT;
  }
  int64_t *dims_tensor1 = aitisa_tensor_dims(tensor1);
  int64_t *dims_tensor2 = aitisa_tensor_dims(tensor2);
  for (int i = 0; i < ndim_tensor1; i++) {
    if (dims_tensor1[i] != dims_tensor2[i]) {
      return STATUS_INVALID_ARGUMENT;
    }
  }
  // create output
  CHECK_STATUS(aitisa_create(aitisa_tensor_data_type(tensor1),
                             aitisa_tensor_device(tensor1),
                             dims_tensor1, ndim_tensor1, NULL, 0, output));
  int64_t size = aitisa_tensor_size(tensor1);
  void *data_tensor1 = aitisa_tensor_data(tensor1);
  void *data_tensor2 = aitisa_tensor_data(tensor2);
  void *data_output = aitisa_tensor_data(*output);
  DataType dtype = aitisa_tensor_data_type(tensor1);
  void *a = malloc(dtype.size), *b = malloc(dtype.size),
       *c = malloc(dtype.size);
  for (int i = 0; i < size; i++) {
    aitisa_get_typed_array_value_func(dtype)(data_tensor1, i, a);
    aitisa_get_typed_array_value_func(dtype)(data_tensor2, i, b);
    aitisa_binary_op_func(dtype)(a, b, OP_DIV, c);
    aitisa_set_typed_array_value_func(dtype)(data_output, i, c);
  }
  free(a);
  free(b);
  free(c);
  return STATUS_SUCCESS;
}
