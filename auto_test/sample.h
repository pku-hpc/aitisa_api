#pragma once

extern "C"{
#include <stdint.h>
}
#include <vector>
#include <cstring>
#include "basic.h"

namespace aitisa_api {

class Unary_Input {
public:
  Unary_Input() { count_ = new int; *count_ = 1; }
  Unary_Input(int64_t ndim, int64_t *dims, int dtype, int device, 
               void *data, unsigned int len);
  Unary_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, 
               int device, void *data, unsigned int len);
  Unary_Input(Unary_Input & input);
  Unary_Input(Unary_Input && input);
  virtual ~Unary_Input() { 
    if(count_){
      (*count_)--;
      if(*count_ == 0){
        delete [] (char*)dims_; 
        delete [] (char*)data_; 
        delete count_;
      }
    } 
  }
  Unary_Input & operator=(Unary_Input& right);
  void to_CUDA() { device_ = 1; }
  void to_CPU() { device_ = 0; }
  int64_t ndim() { return ndim_; }
  int64_t* dims() { return dims_; }
  int dtype() { return dtype_; }
  int device() { return device_; }
  void* data() { return data_; }
  unsigned int len() { return len_; }
  void set_data(void *data, unsigned int len) { data_ = data; len_ = len; }
  int * count() { return count_; }
  void to_nullptr() { count_ = nullptr; dims_ = nullptr; data_ = nullptr; }
private:
  int64_t ndim_ = 0;
  int64_t *dims_ = nullptr;
  int dtype_ = 0;
  int device_ = 0;
  void *data_ = nullptr;
  unsigned int len_ = 0;
  int *count_ = nullptr;
};

class Binary_Input {
public:
  Binary_Input() { count_ = new int; *count_ = 1; }
  Binary_Input(int64_t ndim1, int64_t *dims1, int dtype1, 
                int device1, void *data1, unsigned int len1, 
                int64_t ndim2, int64_t *dims2, int dtype2, 
                int device2, void *data2, unsigned int len2);
  Binary_Input(int64_t ndim1,  std::vector<int64_t> dims1, int dtype1, 
                int device1,  void *data1, unsigned int len1, 
                int64_t ndim2, std::vector<int64_t> dims2, int dtype2, 
                int device2, void *data2, unsigned int len2);
  Binary_Input(Binary_Input & input);
  Binary_Input(Binary_Input && input);
  virtual ~Binary_Input() {
    if(count_){
      (*count_)--;
      if(*count_ == 0){
        delete [] (char*)dims1_;
        delete [] (char*)dims2_;
        delete [] (char*)data1_;
        delete [] (char*)data2_;
        delete count_;
      }
    }  
  }
  Binary_Input & operator=(Binary_Input& right);
  void to_CUDA() { device1_ = 1; device2_ = 1; }
  void to_CPU() { device1_ = 0; device2_ = 0; }
  int64_t ndim1() { return ndim1_; }
  int64_t ndim2() { return ndim2_; }
  int64_t* dims1() { return dims1_; }
  int64_t* dims2() { return dims2_; }
  int dtype1() { return dtype1_; }
  int dtype2() { return dtype2_; }
  int device1() { return device1_; }
  int device2() { return device2_; }
  void* data1() { return data1_; }
  void* data2() { return data2_; }
  unsigned int len1() { return len1_; }
  unsigned int len2() { return len2_; }
  void set_data1(void *data, unsigned int len) { data1_ = data; len1_ = len; }
  void set_data2(void *data, unsigned int len) { data2_ = data; len2_ = len; }
  int * count() { return count_; }
  void to_nullptr() { 
    count_ = nullptr; 
    dims1_ = nullptr;
    dims2_ = nullptr;
    data1_ = nullptr; 
    data2_ = nullptr; 
  }
private:
  int64_t ndim1_ = 0;
  int64_t *dims1_ = nullptr;
  int dtype1_ = 0;
  int device1_ = 0;
  void *data1_ = nullptr;
  unsigned int len1_ = 0;
  int64_t ndim2_ = 0;
  int64_t *dims2_ = nullptr;
  int dtype2_ = 0;
  int device2_ = 0;
  void *data2_ = nullptr;
  unsigned int len2_ = 0;
  int *count_ = nullptr;
};

class Result {
public:
  Result() { count_ = new int; *count_ = 1;}
  Result(Result & result);
  Result(Result && result);
  virtual ~Result() {
    if(count_){
      (*count_)--;
      if(*count_ == 0){
        delete [] dims_;
        delete [] (char*)data_;
      }
    }
  }
  int dtype() const { return dtype_; }
  int device() const { return device_; }
  int64_t ndim() const { return ndim_; }
  int64_t* dims() const { return dims_; }
  void* data() const { return data_; }
  unsigned int len() const { return len_; }
  int * count() { return count_; }
  virtual void set_result(int64_t ndim, int64_t *dims, int dtype, 
                  int device, void *data, unsigned int len);
  virtual void to_nullptr() {
    dims_ = nullptr;
    data_ = nullptr;
    count_ = nullptr;
  }
private:
  int64_t ndim_ = 0;
  int64_t *dims_ = nullptr;
  int dtype_ = 0;
  int device_ = 0;
  void *data_ = nullptr;
  unsigned int len_ = 0;
  int *count_ = nullptr;
};

template <typename InputType>
class Sample {
public:
  Sample() {}
  Sample(InputType& in):input_(in), result_() {}
  Sample(Sample<InputType>& sample): input_(sample.input()), result_(sample.result()) {}
  Sample(Sample<InputType> && sample): input_(sample.input()), result_(sample.result()) {}
  virtual ~Sample() {}
  Sample & operator=(Sample& right){
    input_ = right.input();
    result_ = right.result();
    return *this;
  }
  void set_input(InputType& in) { input_ = in; }
  InputType & input() { return input_; }
  Result & result() { return result_; }
  void set_result(int64_t ndim, int64_t *dims, int dtype, 
                  int device ,void *data, unsigned int len){
    result_.set_result(ndim, dims, dtype, device, data, len);
  }
private:
  InputType input_;
  Result result_;
};

// To make child class of ::testing::Test a concrete class to be instantiated
template <typename T>
class Concrete : public T {
public:
  virtual void TestBody() {}
};

template <typename TestType>
Sample<Binary_Input> get_binary_sample(int sample_num){
  Sample<Binary_Input> sample;
  Concrete<TestType> test;
  int64_t ndim_out, *dims_out;
  void *data_out;
  unsigned int len_out;
  int dtype_out_num, device_out_num;
  AITISA_DataType dtype1, dtype2, dtype_out;
  AITISA_Device device1, device2, device_out;
  AITISA_Tensor tensor_in1, tensor_in2, tensor_out;
  sample.set_input(*(test.input[sample_num]));
  Binary_Input& input = sample.input();
  dtype1 = aitisa_int_to_dtype(input.dtype1());
  dtype2 = aitisa_int_to_dtype(input.dtype2());
  device1 = aitisa_int_to_device(input.device1());
  device2 = aitisa_int_to_device(input.device2());
  aitisa_create(dtype1, device1, input.dims1(), input.ndim1(), 
                input.data1(), input.len1(), &tensor_in1);
  aitisa_create(dtype2, device2, input.dims2(), input.ndim2(), 
                input.data2(), input.len2(), &tensor_in2);
  TestType::aitisa_kernel(tensor_in1, tensor_in2, &tensor_out);
  aitisa_resolve(tensor_out, &dtype_out, &device_out, 
                 &dims_out, &ndim_out, &data_out, &len_out);
  dtype_out_num = aitisa_dtype_to_int(dtype_out);
  device_out_num = aitisa_device_to_int(device_out);
  sample.set_result(ndim_out, dims_out, dtype_out_num, 
                    device_out_num, data_out, len_out);
  aitisa_destroy(&tensor_in1);
  aitisa_destroy(&tensor_in2);
  aitisa_destroy(&tensor_out);
  return sample;
}

// template <typename TestType>
// void get_binary_sample(Sample<Binary_Input>& sample, int sample_num){
//   Concrete<TestType> test;
//   int64_t ndim_out, *dims_out;
//   void *data_out;
//   unsigned int len_out;
//   int dtype_out_num, device_out_num;
//   AITISA_DataType dtype1, dtype2, dtype_out;
//   AITISA_Device device1, device2, device_out;
//   AITISA_Tensor tensor_in1, tensor_in2, tensor_out;
//   sample.set_input(*(test.input[sample_num]));
//   Binary_Input& input = sample.input();
//   dtype1 = aitisa_int_to_dtype(input.dtype1());
//   dtype2 = aitisa_int_to_dtype(input.dtype2());
//   device1 = aitisa_int_to_device(input.device1());
//   device2 = aitisa_int_to_device(input.device2());
//   aitisa_create(dtype1, device1, input.dims1(), input.ndim1(), 
//                 input.data1(), input.len1(), &tensor_in1);
//   aitisa_create(dtype2, device2, input.dims2(), input.ndim2(), 
//                 input.data2(), input.len2(), &tensor_in2);
//   TestType::aitisa_kernel(tensor_in1, tensor_in2, &tensor_out);
//   aitisa_resolve(tensor_out, &dtype_out, &device_out, 
//                  &dims_out, &ndim_out, &data_out, &len_out);
//   dtype_out_num = aitisa_dtype_to_int(dtype_out);
//   device_out_num = aitisa_device_to_int(device_out);
//   sample.set_result(ndim_out, dims_out, dtype_out_num, 
//                     device_out_num, data_out, len_out);
//   aitisa_destroy(&tensor_in1);
//   aitisa_destroy(&tensor_in2);
//   aitisa_destroy(&tensor_out);
// }

} //namespace aitisa_api