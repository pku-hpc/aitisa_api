#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef double Scalar;

typedef enum {
  TYPE_INT8 = 0U,
  TYPE_UINT8,
  TYPE_INT16,
  TYPE_UINT16,
  TYPE_INT32,
  TYPE_UINT32,
  TYPE_INT64,
  TYPE_UINT64,
  TYPE_FLOAT,
  TYPE_DOUBLE,
  TYPE_NTYPES = 10U
} TypeCode;

typedef struct {
  TypeCode code;
  uint8_t size;
} DataType;

typedef void (*GetTypedValueFunc)(void *addr, void *value);
typedef void (*SetTypedValueFunc)(void *addr, void *value);
typedef void (*CasttoTypedValueFunc)(void *addr, double *value);

GetTypedValueFunc aitisa_get_typed_value_func(DataType dtype);
SetTypedValueFunc aitisa_set_typed_value_func(DataType dtype);
CasttoTypedValueFunc aitisa_castto_typed_value_func(DataType dtype);


#endif