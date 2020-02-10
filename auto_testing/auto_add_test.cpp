#include <iostream>
#include "auto_add_test.h"

namespace hice {
  class Tensor {};
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
  } DataType;

  typedef enum {
    kCPU = 0U,
    kCUDA,
    NDEVICES = 2U
  } Device;

  typedef enum {
    DENSE = 0U,
    SPARSE,
    NLAYOUTS = 2U
  } LayoutType;

  void create(int dtype, int device, int layout, int64_t *dims, int64_t ndim, void* data, Tensor* tensor){
    std::cout<< "This is hice create function." << std::endl;
  }
  void resolve(Tensor tensor, int *dtype, int64_t* ndim, int64_t* dims_ptr, void* data){
    std::cout<< "This is hice resolve function." << std::endl;
  }
  void add(Tensor tensor1, Tensor tensor2, Tensor* result){
    std::cout<< "This is hice add function." << std::endl;
  }
}
REGISTER_TENSOR(hice::Tensor, hice::DataType, hice::Device, hice::LayoutType, hice::create, hice::resolve);
REGISTER_ADD(hice::add);
