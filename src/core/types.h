#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

/**
 * @brief Supported Type code used by DataType
 */
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

/**
 * @brief DataType hold by the tensor
 */
typedef struct {
  TypeCode code; /**< TypeCode used by this DataType*/
  uint8_t size;  /**< Number of bytes occupyied by this DataType */
} DataType;

typedef void (*GetTypedValueFunc)(void *addr, void *value);
typedef void (*GetTypedArrayValueFunc)(void *addr, int64_t i, void *value);
typedef void (*SetTypedValueFunc)(void *addr, void *value);
typedef void (*SetTypedArrayValueFunc)(void *addr, int64_t i, void *value);
typedef void (*CasttoTypedValueFunc)(void *addr, double *value);

GetTypedValueFunc aitisa_get_typed_value_func(DataType dtype);
GetTypedArrayValueFunc aitisa_get_typed_array_value_func(DataType dtype);
SetTypedValueFunc aitisa_set_typed_value_func(DataType dtype);
SetTypedArrayValueFunc aitisa_set_typed_array_value_func(DataType dtype);
CasttoTypedValueFunc aitisa_castto_typed_value_func(DataType dtype);

#endif