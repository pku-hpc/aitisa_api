#include <stdint.h>
#include <vector>
#include <cstring>

namespace aitisa_api {

class Unary_Input {
public:
  Unary_Input() {}
  Unary_Input(int64_t ndim, int64_t *dims, int dtype, int device, 
               void *data, unsigned int len);
  Unary_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, 
               int device, void *data, unsigned int len);
  Unary_Input(Unary_Input& sample);
  virtual ~Unary_Input() { delete [] (char*)dims_; delete [] (char*)data_; }
  Unary_Input & operator=(Unary_Input& right);
  int64_t ndim() { return ndim_; }
  int64_t* dims() { return dims_; }
  int dtype() { return dtype_; }
  int device() { return device_; }
  void* data() { return data_; }
  unsigned int len() { return len_; }
  void set_data(void *data, unsigned int len) { data_ = data; len_ = len; }
private:
  int64_t ndim_ = 0;
  int64_t *dims_ = nullptr;
  int dtype_ = 0;
  int device_ = 0;
  void *data_ = nullptr;
  unsigned int len_ = 0;
};

class Binary_Input {
public:
  Binary_Input() {}
  Binary_Input(int64_t ndim1, int64_t *dims1, int dtype1, 
                int device1, void *data1, unsigned int len1, 
                int64_t ndim2, int64_t *dims2, int dtype2, 
                int device2, void *data2, unsigned int len2);
  Binary_Input(int64_t ndim1,  std::vector<int64_t> dims1, int dtype1, 
                int device1,  void *data1, unsigned int len1, 
                int64_t ndim2, std::vector<int64_t> dims2, int dtype2, 
                int device2, void *data2, unsigned int len2);
  Binary_Input(Binary_Input& sample);
  virtual ~Binary_Input() {
    delete [] (char*)dims1_;
    delete [] (char*)dims2_;
    delete [] (char*)data1_;
    delete [] (char*)data2_;
  }
  Binary_Input & operator=(Binary_Input& right);
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
};

class Result {
public:
  Result() {}
  virtual ~Result() {
    delete [] dims_;
    delete [] (char*)data_;
  }
  int dtype() const { return dtype_; }
  int device() const { return device_; }
  int64_t ndim() const { return ndim_; }
  int64_t* dims() const { return dims_; }
  void* data() const { return data_; }
  unsigned int len() { return len_; }
  void set_result(int64_t ndim, int64_t *dims, void *data, unsigned int len) {
    ndim_ = ndim;
    dims_ = dims;
    data_ = data;
    len_ = len;
  }
private:
  int64_t ndim_ = 0;
  int64_t *dims_ = nullptr;
  int dtype_ = 0;
  int device_ = 0;
  void *data_ = nullptr;
  unsigned int len_ = 0;
};

template <typename InputType>
class Sample {
public:
  Sample():input_(), result_() {}
  Sample(InputType& in):input_(in), result_() {}
  Sample(Sample& sample): input_(sample.input()), result_(sample.result()){}
  virtual ~Sample() {}
  Sample & operator=(Sample& right){
    input_ = right.input();
    result_ = right.result();
    return *this;
  }
  void set_input(InputType& in) { input_ = in; }
  InputType & input() { return input_; }
  Result & result() { return result_; }
  void set_result(int64_t ndim, int64_t *dims, void *data, unsigned int len){
    result_.set_result(ndim, dims, data, len);
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

} //namespace aitisa_api