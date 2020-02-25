#include "auto_test/sample.h"
#include <cstring>

namespace aitisa_api {

Unary_Input::Unary_Input(int64_t ndim, int64_t *dims, int dtype, int device, 
                           void *data, unsigned int len): 
  ndim_(ndim), dims_(dims), dtype_(dtype), device_(device), data_(data), len_(len) {}

Unary_Input::Unary_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, 
                           int device, void *data, unsigned int len):
  ndim_(ndim), dims_(nullptr),dtype_(dtype), device_(device), data_(data), len_(len) {
  dims_ = new int64_t[ndim];
  for(int64_t i=0; i<ndim; i++){
    dims_[i] = dims[i];
  }
}

Unary_Input::Unary_Input(Unary_Input& input): 
  ndim_(input.ndim()), dims_(nullptr), dtype_(input.dtype()), 
  device_(input.device()), data_(nullptr), len_(input.len()) {
  dims_ = new int64_t[ndim_];
  data_ = (void*) new char[len_];
  for(int64_t i=0; i<ndim_; i++){
    dims_[i] = input.dims()[i];
  }
  memcpy(data_, input.data(), len_);
}

Unary_Input & Unary_Input::operator=(Unary_Input& right){
  this->ndim_ = right.ndim();
  this->dims_ = new int64_t[this->ndim_];
  this->dtype_ = right.dtype();
  this->device_ = right.device();
  this->len_ = right.len();
  this->data_ = (void*) new char[this->len_];
  for(int64_t i=0; i<ndim_; i++){
    this->dims_[i] = right.dims()[i];
  }
  memcpy(this->data_, right.data(), this->len_);
  return *this;
}

Binary_Input::Binary_Input(int64_t ndim1, int64_t *dims1, int dtype1, 
                             int device1, void *data1, unsigned int len1, 
                             int64_t ndim2, int64_t *dims2, int dtype2, 
                             int device2, void *data2, unsigned int len2): 
  ndim1_(ndim1), dims1_(dims1), dtype1_(dtype1), device1_(device1), data1_(data1), len1_(len1), 
  ndim2_(ndim2), dims2_(dims2), dtype2_(dtype2), device2_(device2), data2_(data2), len2_(len2) {}

Binary_Input::Binary_Input(int64_t ndim1,  std::vector<int64_t> dims1, int dtype1, 
                             int device1,  void *data1, unsigned int len1, 
                             int64_t ndim2, std::vector<int64_t> dims2, int dtype2, 
                             int device2, void *data2, unsigned int len2): 
  ndim1_(ndim1), dims1_(nullptr), dtype1_(dtype1), device1_(device1), data1_(data1), len1_(len1), 
  ndim2_(ndim2), dims2_(nullptr), dtype2_(dtype2), device2_(device2), data2_(data2), len2_(len2) {
  dims1_ = new int64_t[ndim1];
  dims2_ = new int64_t[ndim2];
  for(int64_t i=0; i<ndim1; i++){
    dims1_[i] = dims1[i];
  }
  for(int64_t i=0; i<ndim2; i++){
    dims2_[i] = dims2[i];
  }
}

Binary_Input::Binary_Input(Binary_Input& Input): 
  ndim1_(Input.ndim1()),  dims1_(nullptr), dtype1_(Input.dtype1()), 
  device1_(Input.device1()), data1_(nullptr), len1_(Input.len1()), 
  ndim2_(Input.ndim2()), dims2_(nullptr), dtype2_(Input.dtype2()), 
  device2_(Input.device2()), data2_(nullptr), len2_(Input.len2()) {
  dims1_ = new int64_t[ndim1_];
  dims2_ = new int64_t[ndim2_];
  data1_ = (void*) new char[len1_];
  data2_ = (void*) new char[len2_];
  for(int64_t i=0; i<ndim1_; i++){
    dims1_[i] = Input.dims1()[i];
  }
  for(int64_t i=0; i<ndim2_; i++){
    dims2_[i] = Input.dims2()[i];
  }
  memcpy(data1_, Input.data1(), len1_);
  memcpy(data2_, Input.data2(), len2_);
}

Binary_Input & Binary_Input::operator=(Binary_Input& right){
  this->ndim1_ = right.ndim1();
  this->ndim2_ = right.ndim2();
  this->dims1_ = new int64_t[this->ndim1_];
  this->dims2_ = new int64_t[this->ndim2_];
  this->dtype1_ = right.dtype1();
  this->dtype2_ = right.dtype2();
  this->device1_ = right.device1();
  this->device2_ = right.device2();
  this->len1_ = right.len1();
  this->len2_ = right.len2();
  this->data1_ = (void*) new char[this->len1_];
  this->data2_ = (void*) new char[this->len2_];
  for(int64_t i=0; i<ndim1_; i++){
    this->dims1_[i] = right.dims1()[i];
  }
  for(int64_t i=0; i<ndim2_; i++){
    this->dims2_[i] = right.dims2()[i];
  }
  memcpy(this->data1_, right.data1(), this->len1_);
  memcpy(this->data2_, right.data2(), this->len2_);
  return *this;
}

} // namespace aitisa_api