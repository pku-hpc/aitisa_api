#include <vector>
#include <iostream>
extern "C" {
#include "src/core/tensor.h"
}

DataType dtypes[10] = {kInt8,   kUint8, 
                       kInt16,  kUint16, 
                       kInt32,  kUint32, 
                       kInt64,  kUint64, 
                       kFloat,  kDouble};
Device devices[2] = { {DEVICE_CPU,  0}, 
                      {DEVICE_CUDA, 0} };

int64_t default_ndim = 0;
int64_t *default_dims = NULL;
int default_dtype = 0;
int default_device = 0;
void *default_data = NULL;
int64_t default_len = 0;

#define TRANSLATE_DATA_TYPE(INT_TYPE, AITISA_TYPE)      \
  DataType AITISA_TYPE = dtypes[INT_TYPE];

#define TRANSLATE_DEVICE_TYPE(INT_TYPE, AITISA_TYPE)    \
  Device AITISA_TYPE = devices[INT_TYPE];

#define REGISTER_TENSOR(TENSOR_TYPE, DATA_TYPE, DEVICE_TYPE, CREATE_FUNC, RESOLVE_FUNC)                 \
  class TensorBase {                                                                                    \
  public:                                                                                               \
    using UserTensor = TENSOR_TYPE;                                                                     \
    using UserDataType = DATA_TYPE;                                                                     \
    using UserDeviceType = DEVICE_TYPE;                                                                 \
    static void create(int dtype, int device, int64_t *dims, int64_t ndim,                              \
                       void *data, unsigned int len, UserTensor *tensor){                               \
      CREATE_FUNC(dtype, device, dims, ndim, data, len, tensor);                                        \
    }                                                                                                   \
    static void resolve(UserTensor tensor, int *dtype, int *device,                                     \
                        int64_t **dims, int64_t *ndim, void **data, int64_t *len){                      \
      RESOLVE_FUNC(tensor, dtype, device, dims, ndim, data, len);                                       \
    }                                                                                                   \
  };

#define natural_assign_int_case(INT_DTYPE, REAL_DTYPE)        \
  case INT_DTYPE: {                                           \
    for(int64_t i=0; i<len/sizeof(REAL_DTYPE); i++){          \
      ((REAL_DTYPE*)data)[i] = i;                             \
    }                                                         \
    break;                                                    \
  }
#define natural_assign_float_case(INT_DTYPE, REAL_DTYPE)      \
  case INT_DTYPE: {                                           \
    for(int64_t i=0; i<len/sizeof(REAL_DTYPE); i++){          \
      ((REAL_DTYPE*)data)[i] = i * 0.1;                       \
    }                                                         \
    break;                                                    \
  }

void natural_assign(void *data, int64_t len, int dtype){
  switch (dtype) {
    natural_assign_int_case(0, int8_t);
    natural_assign_int_case(1, uint8_t);
    natural_assign_int_case(2, int16_t);
    natural_assign_int_case(3, uint16_t);
    natural_assign_int_case(4, int32_t);
    natural_assign_int_case(5, uint32_t);
    natural_assign_int_case(6, int64_t);
    natural_assign_int_case(7, uint64_t);
    natural_assign_float_case(8, float);
    natural_assign_float_case(9, double);
    default: break;
  }
}

template <typename DATATYPE>
void natural_assign_int(DATATYPE *data, int64_t nelem){
  for(int64_t i=0; i<nelem; i++){
    data[i] = (DATATYPE)i;
  }
}

template <typename DATATYPE>
void natural_assign_float(DATATYPE *data, int64_t nelem){
  for(int64_t i=0; i<nelem; i++){
    data[i] = (DATATYPE)(i * 0.1);
  }
}

template <typename DATA_TYPE>
class Unary_Sample {
public:
  Unary_Sample(int64_t ndim=default_ndim, int64_t *dims=default_dims, 
                int dtype=default_dtype, int device=default_device, 
                DATA_TYPE *data=default_data, int64_t len=default_len): 
                ndim_(ndim), dims_(dims), dtype_(dtype), 
                device_(device),  data_(data), len_(len) {}
  Unary_Sample(int64_t ndim, std::vector<int64_t> dims, int dtype, 
               int device, DATA_TYPE *data, int64_t len) {
    ndim_ = ndim;
    dims_ = new int64_t[ndim];
    dtype_ = dtype;
    device_ = device;
    data_ = data;
    for(int64_t i=0; i<ndim; i++){
      dims_[i] = dims[i];
    }
  }
  ~Unary_Sample() {
    delete [] dims_;
    delete [] data_;
  }

  int64_t ndim_;
  int64_t *dims_;
  int dtype_;
  int device_;
  DATA_TYPE *data_;
  int64_t len_;
};

template <typename DATA_TYPE>
class Binary_Sample {
public:
  Binary_Sample(int64_t ndim1=default_ndim, int64_t ndim2=default_ndim,
                int64_t *dims1=default_dims, int64_t *dims2=default_dims, 
                int dtype1=default_dtype, int dtype2=default_dtype,
                int device1=default_device, int device2=default_device,
                DATA_TYPE *data1=default_data, DATA_TYPE *data2=default_data, 
                int64_t len1=default_len, int64_t len2=default_len): 
                ndim1_(ndim1), ndim2_(ndim2), dims1_(dims1), dims2_(dims2), 
                dtype1_(dtype1), dtype2_(dtype2), device1_(device1), device2_(device2),
                data1_(data1), data2_(data2), len1_(len1), len2_(len2) {}
  Binary_Sample(int64_t ndim1, int64_t ndim2, std::vector<int64_t> dims1, std::vector<int64_t> dims2, 
                int dtype1, int dtype2, int device1, int device2, DATA_TYPE *data1, DATA_TYPE *data2, 
                int64_t len1, int64_t len2) {
    ndim1_ = ndim1;
    ndim2_ = ndim2;
    dims1_ = new int64_t[ndim1];
    dims2_ = new int64_t[ndim2];
    dtype1_ = dtype1;
    dtype2_ = dtype2;
    device1_ = device1;
    device2_ = device2;
    data1_ = data1;
    data2_ = data2;
    for(int64_t i=0; i<ndim1; i++){
      dims1_[i] = dims1[i];
    }
    for(int64_t i=0; i<ndim2; i++){
      dims2_[i] = dims2[i];
    }
  }
  ~Binary_Sample() {
    delete [] dims1_;
    delete [] dims2_;
    delete [] data1_;
    delete [] data2_;
  }

  int64_t ndim1_;
  int64_t *dims1_;
  int dtype1_;
  int device1_;
  DATA_TYPE *data1_;
  int64_t len1_;

  int64_t ndim2_;
  int64_t *dims2_;
  int dtype2_;
  int device2_;
  DATA_TYPE *data2_;
  int64_t len2_;
};

