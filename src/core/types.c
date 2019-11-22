#include "src/core/types.h"

#include <stdint.h>

DataType type_int8 = {TYPE_INT8, sizeof(int8_t)};
DataType type_uint8 = {TYPE_UINT8, sizeof(uint8_t)};

DataType type_int16 = {TYPE_INT16, sizeof(int16_t)};
DataType type_uint16 = {TYPE_UINT16, sizeof(uint16_t)};

DataType type_int32 = {TYPE_INT32, sizeof(int32_t)};
DataType type_uint32 = {TYPE_UINT32, sizeof(uint32_t)};

DataType type_int64 = {TYPE_INT64, sizeof(int64_t)};
DataType type_uint64 = {TYPE_UINT64, sizeof(uint64_t)};

DataType type_float = {TYPE_FLOAT, sizeof(float)};
DataType type_double = {TYPE_UINT64, sizeof(double)};

void get_int8_value(void *addr, void *value) {
  *(int8_t *)value = *(int8_t *)addr;
}

void get_uint8_value(void *addr, void *value) {
  *(uint8_t *)value = *(uint8_t *)addr;
}

void get_int16_value(void *addr, void *value) {
  *(int16_t *)value = *(int16_t *)addr;
}

void get_uint16_value(void *addr, void *value) {
  *(uint16_t *)value = *(uint16_t *)addr;
}

void get_int32_value(void *addr, void *value) {
  *(int32_t *)value = *(int32_t *)addr;
}

void get_uint32_value(void *addr, void *value) {
  *(uint32_t *)value = *(uint32_t *)addr;
}

void get_int64_value(void *addr, void *value) {
  *(int64_t *)value = *(int64_t *)addr;
}

void get_uint64_value(void *addr, void *value) {
  *(uint64_t *)value = *(uint64_t *)addr;
}

void get_float_value(void *addr, void *value) {
  *(float *)value = *(float *)addr;
}

void get_double_value(void *addr, void *value) {
  *(double *)value = *(double *)addr;
}

void set_int8_value(void *addr, void *value) {
  *(int8_t *)addr = *(int8_t *)value;
}

void set_uint8_value(void *addr, void *value) {
  *(uint8_t *)addr = *(uint8_t *)value;
}

void set_int16_value(void *addr, void *value) {
  *(int16_t *)addr = *(int16_t *)value;
}

void set_uint16_value(void *addr, void *value) {
  *(uint16_t *)addr = *(uint16_t *)value;
}

void set_int32_value(void *addr, void *value) {
  *(int32_t *)addr = *(int32_t *)value;
}

void set_uint32_value(void *addr, void *value) {
  *(uint32_t *)addr = *(uint32_t *)value;
}

void set_int64_value(void *addr, void *value) {
  *(int64_t *)addr = *(int64_t *)value;
}

void set_uint64_value(void *addr, void *value) {
  *(uint64_t *)addr = *(uint64_t *)value;
}

void set_float_value(void *addr, void *value) {
  *(float *)addr = *(float *)value;
}

void set_double_value(void *addr, void *value) {
  *(double *)addr = *(double *)value;
}

void castto_int8_value(void *addr, double *value) {
  *(int8_t *)addr = *value;
}

void castto_uint8_value(void *addr, double *value) {
  *(uint8_t *)addr = *value;
}

void castto_int16_value(void *addr, double *value) {
  *(int16_t *)addr = *value;
}

void castto_uint16_value(void *addr, double *value) {
  *(uint16_t *)addr = *value;
}

void castto_int32_value(void *addr, double *value) {
  *(int32_t *)addr = *value;
}

void castto_uint32_value(void *addr, double *value) {
  *(uint32_t *)addr = *value;
}

void castto_int64_value(void *addr, double *value) {
  *(int64_t *)addr = *value;
}

void castto_uint64_value(void *addr, double *value) {
  *(uint64_t *)addr = *value;
}

void castto_float_value(void *addr, double *value) {
  *(float *)addr = *value;
}

void castto_double_value(void *addr, double *value) {
  *(double *)addr = *value;
}

void get_int8_array_value(void *addr, int64_t i, void *value) {
  *(int8_t *)value = ((int8_t *)addr)[i];
}

void get_uint8_array_value(void *addr, int64_t i, void *value) {
  *(uint8_t *)value = ((uint8_t *)addr)[i];
}

void get_int16_array_value(void *addr, int64_t i, void *value) {
  *(int16_t *)value = ((int16_t *)addr)[i];
}

void get_uint16_array_value(void *addr, int64_t i, void *value) {
  *(uint16_t *)value = ((uint16_t *)addr)[i];
}

void get_int32_array_value(void *addr, int64_t i, void *value) {
  *(int32_t *)value = ((int32_t *)addr)[i];
}

void get_uint32_array_value(void *addr, int64_t i, void *value) {
  *(uint32_t *)value = ((uint32_t *)addr)[i];
}

void get_int64_array_value(void *addr, int64_t i, void *value) {
  *(int64_t *)value = ((int64_t *)addr)[i];
}

void get_uint64_array_value(void *addr, int64_t i, void *value) {
  *(uint64_t *)value = ((uint64_t *)addr)[i];
}

void get_float_array_value(void *addr, int64_t i, void *value) {
  *(float *)value = ((float *)addr)[i];
}

void get_double_array_value(void *addr, int64_t i, void *value) {
  *(double *)value = ((double *)addr)[i];
}

void set_int8_array_value(void *addr, int64_t i, void *value) {
  ((int8_t *)addr)[i] = *(int8_t *)value;
}

void set_uint8_array_value(void *addr, int64_t i, void *value) {
  ((uint8_t *)addr)[i] = *(uint8_t *)value;
}

void set_int16_array_value(void *addr, int64_t i, void *value) {
  ((int16_t *)addr)[i] = *(int16_t *)value;
}

void set_uint16_array_value(void *addr, int64_t i, void *value) {
  ((uint16_t *)addr)[i] = *(uint16_t *)value;
}

void set_int32_array_value(void *addr, int64_t i, void *value) {
  ((int32_t *)addr)[i] = *(int32_t *)value;
}

void set_uint32_array_value(void *addr, int64_t i, void *value) {
  ((uint32_t *)addr)[i] = *(uint32_t *)value;
}

void set_int64_array_value(void *addr, int64_t i, void *value) {
  ((int64_t *)addr)[i] = *(int64_t *)value;
}

void set_uint64_array_value(void *addr, int64_t i, void *value) {
  ((uint64_t *)addr)[i] = *(uint64_t *)value;
}

void set_float_array_value(void *addr, int64_t i, void *value) {
  ((float *)addr)[i] = *(float *)value;
}

void set_double_array_value(void *addr, int64_t i, void *value) {
  ((double *)addr)[i] = *(double *)value;
}

GetTypedValueFunc get_typed_value_funcs[TYPE_NTYPES] = {
  get_int8_value,
  get_uint8_value,
  get_int16_value,
  get_uint16_value,
  get_int32_value,
  get_uint32_value,
  get_int64_value,
  get_uint64_value,
  get_float_value,
  get_double_value
};

GetTypedArrayValueFunc get_typed_array_value_funcs[TYPE_NTYPES] = {
  get_int8_array_value,
  get_uint8_array_value,
  get_int16_array_value,
  get_uint16_array_value,
  get_int32_array_value,
  get_uint32_array_value,
  get_int64_array_value,
  get_uint64_array_value,
  get_float_array_value,
  get_double_array_value
};

SetTypedValueFunc set_typed_value_funcs[TYPE_NTYPES] = {
  set_int8_value,
  set_uint8_value,
  set_int16_value,
  set_uint16_value,
  set_int32_value,
  set_uint32_value,
  set_int64_value,
  set_uint64_value,
  set_float_value,
  set_double_value
};

SetTypedArrayValueFunc set_typed_array_value_funcs[TYPE_NTYPES] = {
  set_int8_array_value,
  set_uint8_array_value,
  set_int16_array_value,
  set_uint16_array_value,
  set_int32_array_value,
  set_uint32_array_value,
  set_int64_array_value,
  set_uint64_array_value,
  set_float_array_value,
  set_double_array_value
};

CasttoTypedValueFunc castto_typed_value_funcs[TYPE_NTYPES] = {
  castto_int8_value,
  castto_uint8_value,
  castto_int16_value,
  castto_uint16_value,
  castto_int32_value,
  castto_uint32_value,
  castto_int64_value,
  castto_uint64_value,
  castto_float_value,
  castto_double_value
};

GetTypedValueFunc aitisa_get_typed_value_func(DataType dtype) {
  return get_typed_value_funcs[dtype.code];
}

GetTypedArrayValueFunc aitisa_get_typed_array_value_func(DataType dtype) {
  return get_typed_array_value_funcs[dtype.code];
}

SetTypedValueFunc aitisa_set_typed_value_func(DataType dtype) {
  return set_typed_value_funcs[dtype.code];
}

SetTypedArrayValueFunc aitisa_set_typed_array_value_func(DataType dtype) {
  return set_typed_array_value_funcs[dtype.code];
}

CasttoTypedValueFunc aitisa_castto_typed_value_func(DataType dtype) {
  return castto_typed_value_funcs[dtype.code];
}
