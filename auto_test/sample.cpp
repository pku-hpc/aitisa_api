#include "auto_test/sample.h"
#include <cstring>

namespace aitisa_api {

Unary_Input::Unary_Input(int64_t ndim, int64_t *dims, int dtype, int device, 
                         void *data, unsigned int len): 
    ndim_(ndim), dims_(dims), dtype_(dtype), device_(device), data_(data), len_(len) {
  count_ = new int;
  *count_ = 1;
}

Unary_Input::Unary_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, 
                           int device, void *data, unsigned int len):
  ndim_(ndim), dims_(nullptr),dtype_(dtype), device_(device), data_(data), len_(len) {
  dims_ = new int64_t[ndim];
  for(int64_t i=0; i<ndim; i++){
    dims_[i] = dims[i];
  }
  count_ = new int;
  *count_ = 1;
}

Unary_Input::Unary_Input(Unary_Input& input): 
  ndim_(input.ndim()), dims_(input.dims()), dtype_(input.dtype()), 
  device_(input.device()), data_(input.data()), len_(input.len()),
  count_(input.count()) {
  (*count_)++;
}

Unary_Input::Unary_Input(Unary_Input && input): 
  ndim_(input.ndim()), dims_(input.dims()), dtype_(input.dtype()), 
  device_(input.device()), data_(input.data()), len_(input.len()),
  count_(input.count()) {
  input.to_nullptr();
}

Unary_Input & Unary_Input::operator=(Unary_Input& right){
  (*(this->count_))--;
  if(*(this->count_) == 0){
    delete [] this->dims_;
    delete [] (char*)this->data_;
  }
  this->ndim_ = right.ndim();
  this->dims_ = right.dims();
  this->dtype_ = right.dtype();
  this->device_ = right.device();
  this->len_ = right.len();
  this->data_ = right.data();
  this->count_ = right.count();
  (*(this->count_))++;
  return *this;
}

Binary_Input::Binary_Input(int64_t ndim1, int64_t *dims1, int dtype1, 
                           int device1, void *data1, unsigned int len1, 
                           int64_t ndim2, int64_t *dims2, int dtype2, 
                           int device2, void *data2, unsigned int len2): 
    ndim1_(ndim1), dims1_(dims1), dtype1_(dtype1), device1_(device1), data1_(data1), len1_(len1), 
    ndim2_(ndim2), dims2_(dims2), dtype2_(dtype2), device2_(device2), data2_(data2), len2_(len2) {
  count_ = new int;
  *count_ = 1;
}

Binary_Input::Binary_Input(int64_t ndim1, std::vector<int64_t> dims1, int dtype1, 
                           int device1, void *data1, unsigned int len1, 
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
  count_ = new int;
  *count_ = 1;
}

Binary_Input::Binary_Input(Binary_Input & input): 
  ndim1_(input.ndim1()),  dims1_(input.dims1()), dtype1_(input.dtype1()), 
  device1_(input.device1()), data1_(input.data1()), len1_(input.len1()), 
  ndim2_(input.ndim2()), dims2_(input.dims2()), dtype2_(input.dtype2()), 
  device2_(input.device2()), data2_(input.data2()), len2_(input.len2()),
  count_(input.count()) {
  (*count_)++;
}

Binary_Input::Binary_Input(Binary_Input && input): 
  ndim1_(input.ndim1()),  dims1_(input.dims1()), dtype1_(input.dtype1()), 
  device1_(input.device1()), data1_(input.data1()), len1_(input.len1()), 
  ndim2_(input.ndim2()), dims2_(input.dims2()), dtype2_(input.dtype2()), 
  device2_(input.device2()), data2_(input.data2()), len2_(input.len2()),
  count_(input.count()) {
  input.to_nullptr();
}

Binary_Input & Binary_Input::operator=(Binary_Input& right){
  (*(this->count_))--;
  if(*(this->count_) == 0){
    delete [] this->dims1_;
    delete [] this->dims2_;
    delete [] (char*)this->data1_;
    delete [] (char*)this->data2_;
  }
  this->ndim1_ = right.ndim1();
  this->ndim2_ = right.ndim2();
  this->dims1_ = right.dims1();
  this->dims2_ = right.dims2();
  this->dtype1_ = right.dtype1();
  this->dtype2_ = right.dtype2();
  this->device1_ = right.device1();
  this->device2_ = right.device2();
  this->len1_ = right.len1();
  this->len2_ = right.len2();
  this->data1_ = right.data1();
  this->data2_ = right.data2();
  this->count_ = right.count();
  (*(this->count_))++;
  return *this;
}

Result::Result(Result& result):
  ndim_(result.ndim()), dims_(result.dims()), dtype_(result.dtype()),
  device_(result.device()), data_(result.data()), len_(result.len()),
  count_(result.count()){
  (*count_)++;
}

Result::Result(Result && result):
  ndim_(result.ndim()), dims_(result.dims()), dtype_(result.dtype()),
  device_(result.device()), data_(result.data()), len_(result.len()),
  count_(result.count()){
  result.to_nullptr();
}

void Result::set_result(int64_t ndim, int64_t *dims, int dtype, 
                        int device, void *data, unsigned int len) {
  ndim_ = ndim;
  dtype_ = dtype;
  device_ = device;
  len_ = len;
  dims_ = new int64_t[ndim];
  data_ = (void*) new char[len];
  memcpy(dims_, dims, ndim*sizeof(int64_t));
  memcpy(data_, data, len);
}

} // namespace aitisa_api