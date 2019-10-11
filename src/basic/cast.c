#include "src/basic/cast.h"


#define cast_kernel(in_typename, out_typename, in_data, out_data, size)   \
  in_typename *typed_in_data = in_data;                                   \
  out_typename *typed_out_data = out_data;                                \
  for(int64_t i=0; i<size; i++){                                          \
    typed_out_data[i] = (out_typename)(typed_in_data[i]);                 \
  }

static Status cast_template(const Tensor input, DataType dtype,
                            Tensor *output){
  DataType in_dtype = aitisa_tensor_data_type(input);
  int64_t size = aitisa_tensor_size(input);
  void *in_data = aitisa_tensor_data(input);
  void *out_data = aitisa_tensor_data(*output);
  switch(in_dtype.code){
    case TYPE_INT8: {
      switch(dtype.code){
        case TYPE_INT8: {
          // do nothing
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(int8_t, uint8_t, in_data, out_data, size);
        }
        case TYPE_INT16: {
          cast_kernel(int8_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(int8_t, uint16_t, in_data, out_data, size);
        }
        case TYPE_INT32: {
          cast_kernel(int8_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(int8_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(int8_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(int8_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(int8_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(int8_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_UINT8: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(uint8_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          // do nothing
          break;
        }
        case TYPE_INT16: {
          cast_kernel(uint8_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(uint8_t, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(uint8_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(uint8_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(uint8_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(uint8_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(uint8_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(uint8_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_INT16: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(int16_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(int16_t, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          // do nothing
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(int16_t, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(int16_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(int16_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(int16_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(int16_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(int16_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(int16_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_UINT16: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(uint16_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(uint16_t, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(uint16_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          // do nothing
          break;
        }
        case TYPE_INT32: {
          cast_kernel(uint16_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(uint16_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(uint16_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(uint16_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(uint16_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(uint16_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_INT32: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(int32_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(int32_t, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(int32_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(int32_t, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          // do nothing
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(int32_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(int32_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(int32_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(int32_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(int32_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_UINT32: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(uint32_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(uint32_t, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(uint32_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(uint32_t, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(uint32_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          // do nothing
          break;
        }
        case TYPE_INT64: {
          cast_kernel(uint32_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(uint32_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(uint32_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(uint32_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_INT64: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(int64_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(int64_t, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(int64_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(int64_t, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(int64_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(int64_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          // do nothing
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(int64_t, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(int64_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(int64_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_UINT64: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(uint64_t, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(uint64_t, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(uint64_t, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(uint64_t, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(uint64_t, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(uint64_t, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(uint64_t, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          // do nothing
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(uint64_t, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(uint64_t, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_FLOAT: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(float, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(float, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(float, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(float, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(float, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(float, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(float, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(float, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          // do nothing
          break;
        }
        case TYPE_DOUBLE: {
          cast_kernel(float, double, in_data, out_data, size);
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    case TYPE_DOUBLE: {
      switch(dtype.code){
        case TYPE_INT8: {
          cast_kernel(double, int8_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT8: {
          cast_kernel(double, uint8_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT16: {
          cast_kernel(double, int16_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT16: {
          cast_kernel(double, uint16_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT32: {
          cast_kernel(double, int32_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT32: {
          cast_kernel(double, uint32_t, in_data, out_data, size);
          break;
        }
        case TYPE_INT64: {
          cast_kernel(double, int64_t, in_data, out_data, size);
          break;
        }
        case TYPE_UINT64: {
          cast_kernel(double, uint64_t, in_data, out_data, size);
          break;
        }
        case TYPE_FLOAT: {
          cast_kernel(double, float, in_data, out_data, size);
          break;
        }
        case TYPE_DOUBLE: {
          // do nothing
          break;
        }
        default:
          return STATUS_NOT_SUPPORTED;
      }
      break;
    }
    default:
      return STATUS_NOT_SUPPORTED;
  }
  return STATUS_SUCCESS;
}

Status aitisa_cast(const Tensor input, DataType dtype,
                   Tensor *output){
  // if the DataType of input and the DataType to cast to are identical,
  // then do nothing.
  DataType in_dtype = aitisa_tensor_data_type(input);
  if(in_dtype.code == dtype.code){
    return STATUS_SUCCESS;
  }
  // create output
  Tensor new_tensor;
  Device device = aitisa_tensor_device(input);
  LayoutType layout_type = aitisa_tensor_layout_type(input);
  int64_t *out_dims = aitisa_tensor_dims(input);
  int64_t out_ndim = aitisa_tensor_ndim(input);
  CHECK_STATUS(
    aitisa_create(dtype, device, layout_type, out_dims, out_ndim, &new_tensor));
  *output = new_tensor;
  //implement cast
  Status status;
  status = cast_template(input, dtype, output);
  return status;
}
