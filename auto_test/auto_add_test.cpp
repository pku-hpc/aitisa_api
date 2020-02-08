#include <iostream>
#include "auto_test/auto_add_test.h"

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

  void create(int dtype, int device, int64_t *dims, int64_t ndim, 
              void* data, unsigned int len, Tensor* tensor){
    std::cout<< "This is hice create function." << std::endl;
  }
  void resolve(Tensor tensor, int *dtype, int *device, int64_t **dims, int64_t *ndim, void **data, int64_t *len){
    std::cout<< "This is hice resolve function." << std::endl;
  }
  void add(Tensor tensor1, Tensor tensor2, Tensor* result){
    std::cout<< "This is hice add function." << std::endl;
  }
}// namespace hice
REGISTER_TENSOR(hice::Tensor, hice::DataType, hice::Device, hice::create, hice::resolve);
REGISTER_ADD(hice::add);
