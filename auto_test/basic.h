#pragma once

#include <iostream>
#include "gtest/gtest.h"
extern "C" {
#include "src/core/tensor.h"
}
namespace aitisa_api {

extern const DataType aitisa_dtypes[];
extern const Device aitisa_devices[];
inline DataType aitisa_int_to_dtype(int n){ return aitisa_dtypes[n]; }
inline Device aitisa_int_to_device(int n){ return aitisa_devices[n]; }
inline int aitisa_dtype_to_int(DataType dtype){ return static_cast<int>(dtype.code); }
inline int aitisa_device_to_int(Device device){ return static_cast<int>(device.type); }
inline unsigned int elem_size(int dtype) { return static_cast<unsigned int>(aitisa_dtypes[dtype].size); }

void natural_assign(void *data, unsigned int len, int dtype);
void random_assign(void *data, unsigned int len, int dtype);

using AITISA_Tensor = Tensor;
using AITISA_Device = Device;
using AITISA_DataType = DataType;

#define GREEN "\033[32m"
#define RESET "\033[0m"

#define REGISTER_BASIC(TENSOR, DATA_TYPE, INT_TO_DTYPE, DTYPE_TO_INT, DEVICE, INT_TO_DEVICE, DEVICE_TO_INT, CREATE, RESOLVE)  \
  class Basic {                                                                                                               \
  public:                                                                                                                     \
    using UserTensor = TENSOR;                                                                                                \
    using UserDataType = DATA_TYPE;                                                                                           \
    using UserDevice = DEVICE;                                                                                                \
    static UserDataType user_int_to_dtype(int data_type_num){                                                                 \
      return INT_TO_DTYPE(data_type_num);                                                                                     \
    }                                                                                                                         \
    static UserDevice user_int_to_device(int device_type_num){                                                                \
      return INT_TO_DEVICE(device_type_num);                                                                                  \
    }                                                                                                                         \
    static int user_dtype_to_int(UserDataType dtype){                                                                         \
      return DTYPE_TO_INT(dtype);                                                                                             \
    }                                                                                                                         \
    static int user_device_to_int(UserDevice device){                                                                         \
      return DEVICE_TO_INT(device);                                                                                           \
    }                                                                                                                         \
    static void user_create(UserDataType dtype, UserDevice device, int64_t *dims, int64_t ndim,                               \
                       void *data, unsigned int len, UserTensor *tensor){                                                     \
      CREATE(dtype, device, dims, ndim, data, len, tensor);                                                                   \
    }                                                                                                                         \
    static void user_resolve(UserTensor tensor, UserDataType *dtype, UserDevice *device,                                      \
                        int64_t **dims, int64_t *ndim, void **data, unsigned int *len){                                       \
      RESOLVE(tensor, dtype, device, dims, ndim, data, len);                                                                  \
    }                                                                                                                         \
  };


// functions for debug
template <typename T>
void print_data(T* data, unsigned int n){
  for(unsigned int i=0; i<n; i++){
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void print_data2d(T* data, unsigned int m, unsigned int n){
  for(unsigned int i=0; i<m; i++){
    for(unsigned int j=0; j<n; j++){
      std::cout << data[i*n+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
}

} // namespace aitisa_api