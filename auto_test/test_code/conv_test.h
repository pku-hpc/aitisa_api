#pragma once

#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/nn/conv.h"
#include <math.h>
#include <sys/time.h>
}

namespace aitisa_api {

namespace {

class Conv_Input : public Binary_Input {
public:
  Conv_Input() {};
  Conv_Input(int64_t ndim1, int64_t *dims1, int dtype1, 
            int device1, void *data1, unsigned int len1, 
            int64_t ndim2, int64_t *dims2, int dtype2, 
            int device2, void *data2, unsigned int len2,
            int *stride, int *padding, int *dilation, int groups):
    Binary_Input(ndim1, dims1, dtype1, device1, data1, len1,
                 ndim2, dims2, dtype2, device2, data2, len2),
    stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {}
  Conv_Input(int64_t ndim1,  std::vector<int64_t> dims1, int dtype1, 
            int device1,  void *data1, unsigned int len1, 
            int64_t ndim2, std::vector<int64_t> dims2, int dtype2, 
            int device2, void *data2, unsigned int len2,
            std::vector<int> stride, std::vector<int> padding, 
            std::vector<int> dilation, int groups): 
    Binary_Input(ndim1, dims1, dtype1, device1, data1, len1, 
                 ndim2, dims2, dtype2, device2, data2, len2),
    stride_(nullptr), padding_(nullptr), dilation_(nullptr), groups_(groups) {
    int spatial_len = ndim1 - 2;
    this->stride_ = new int[spatial_len];
    this->padding_ = new int[spatial_len];
    this->dilation_ = new int[spatial_len];
    for(int i=0; i<spatial_len; i++){
      this->stride_[i] = stride[i];
      this->padding_[i] = padding[i];
      this->dilation_[i] = dilation[i];
    }
  }
  virtual ~Conv_Input() {
    delete [] stride_;
    delete [] padding_;
    delete [] dilation_;
  }
  Conv_Input & operator=(Conv_Input& right) {
    int spatial_len = right.ndim1() - 2;
    Binary_Input& left = (Binary_Input&)(*this);
    left = (Binary_Input&)right;
    this->stride_ = new int[spatial_len];
    this->padding_ = new int[spatial_len];
    this->dilation_ = new int[spatial_len];
    memcpy(this->stride_, right.stride(), spatial_len*sizeof(int));
    memcpy(this->padding_, right.padding(), spatial_len*sizeof(int));
    memcpy(this->dilation_, right.dilation(), spatial_len*sizeof(int));
  }
  int* stride() { return stride_; }
  int* padding() { return padding_; }
  int* dilation() { return dilation_; }
  int groups() { return groups_; }
private:
  int *stride_ = nullptr;
  int *padding_ = nullptr;
  int *dilation_ = nullptr;
  int groups_ = 1;
};

} // namespace anonymous

template <typename InterfaceType>
class ConvTest : public ::testing::Test{
public:
  ConvTest():
    input0(/*ndim1*/4, /*dims1*/{6,32,124,128}, /*dtype1=float*/8,  
           /*device1=cuda*/1, /*data1*/nullptr, /*len1*/0, 
           /*ndim2*/4, /*dims2*/{64,32,2,2}, /*dtype2=float*/8, 
           /*device2=cuda*/1, /*data2*/nullptr, /*len2*/0,
           /*stride*/{2,2}, /*padding*/{0,0}, /*dilation*/{1,1},
            /*groups*/1),
    input1(/*ndim1*/4, /*dims1*/{7,16,100,180}, /*dtype1=float*/8,  
           /*device1=cuda*/1, /*data1*/nullptr, /*len1*/0, 
           /*ndim2*/4, /*dims2*/{32,16,3,3}, /*dtype2=float*/8, 
           /*device2=cuda*/1, /*data2*/nullptr, /*len2*/0,
           /*stride*/{3,3}, /*padding*/{1,0}, /*dilation*/{2,2},
            /*groups*/1){
    input[0] = &input0;
    input[1] = &input1;
    ninput = 2;
    for(int i=0; i<ninput; i++){
      unsigned int input_nelem1 = 1;
      unsigned int input_nelem2 = 1;
      for(unsigned int j=0; j<input[i]->ndim1(); j++){
        input_nelem1 *= input[i]->dims1()[j];
      }
      for(unsigned int j=0; j<input[i]->ndim2(); j++){
        input_nelem2 *= input[i]->dims2()[j];
      }
      unsigned int input_len1 = input_nelem1 * elem_size(input[i]->dtype1());
      unsigned int input_len2 = input_nelem2 * elem_size(input[i]->dtype2());
      void *input_data1 = (void*) new char[input_len1];
      void *input_data2 = (void*) new char[input_len2];
      random_assign(input_data1, input_len1, input[i]->dtype1());
      random_assign(input_data2, input_len2, input[i]->dtype2());
      input[i]->set_data1(input_data1, input_len1);
      input[i]->set_data2(input_data2, input_len2);
      //.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // print_data2d((float*)input_data1, 4, 4);
      // print_data2d((float*)input_data2, 2, 2);
    }
  }
  virtual ~ConvTest(){}
  using InputType = Conv_Input;
  using UserInterface = InterfaceType;
  static void aitisa_kernel(const AITISA_Tensor input, const AITISA_Tensor filter, 
                            int *stride, const int *padding, const int *dilation, 
                            const int groups, AITISA_Tensor *output){
    aitisa_conv(input, filter, stride, padding, dilation, groups, output);
  }
  // inputs
  Conv_Input input0; // Natural assigned int32 type input of CPU with InputDims1{3,3,10,6}, FilterDims2{5,3,2,2}, stride{2,2}, padding{0,0}, dilation{1,1}
  Conv_Input input1; // Random assigned double type input of CUDA with InputDims1{10,3,100,124,20}, FilterDims2{10,3,5,5,5}, stride{5,5,5}, padding{0,1,0}, dilation{1,1,1}
  Conv_Input *input[2] = {&input0, &input1};
  std::string input0_name = "Random float of CUDA with InputDims{6,32,124,128}, FilterDims{64,32,2,2}, stride{2,2}, padding{0,0}, dilation{1,1}";
  std::string input1_name = "Random float of CUDA with InputDims{7,16,100,100}, FilterDims{32,16,3,3}, stride{3,3}, padding{1,0}, dilation{2,2}";
  std::string *input_name[2] = {&input0_name, &input1_name};
  int ninput = 2;
};
TYPED_TEST_CASE_P(ConvTest);

TYPED_TEST_P(ConvTest, TwoTests){
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;
  for(int i=0; i<this->ninput; i++){
    // if(i==0) continue;
    struct timeval aitisa_start, aitisa_end, user_start, user_end;
    double aitisa_time, user_time;
    int64_t aitisa_result_ndim, user_result_ndim;
    int64_t *aitisa_result_dims=nullptr, *user_result_dims=nullptr;
    float *aitisa_result_data=nullptr, *user_result_data=nullptr;
    unsigned int aitisa_result_len, user_result_len;
    AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
    AITISA_DataType aitisa_result_dtype;
    AITISA_Device aitisa_result_device;
    UserTensor user_tensor1, user_tensor2, user_result;
    UserDataType user_result_dtype;
    UserDevice user_result_device;
    // aitisa
    AITISA_DataType aitisa_dtype1 = aitisa_int_to_dtype(this->input[i]->dtype1());
    AITISA_DataType aitisa_dtype2 = aitisa_int_to_dtype(this->input[i]->dtype2());
    AITISA_Device aitisa_device1 = aitisa_int_to_device(0); // cpu supoorted only
    AITISA_Device aitisa_device2 = aitisa_int_to_device(0); // cpu supported only
    aitisa_create(aitisa_dtype1, aitisa_device1, this->input[i]->dims1(), this->input[i]->ndim1(), 
                  (void*)(this->input[i]->data1()), this->input[i]->len1(), &aitisa_tensor1);
    aitisa_create(aitisa_dtype2, aitisa_device2, this->input[i]->dims2(), this->input[i]->ndim2(), 
                  (void*)(this->input[i]->data2()), this->input[i]->len2(), &aitisa_tensor2);
    gettimeofday(&aitisa_start,NULL); 
    aitisa_conv(aitisa_tensor1, aitisa_tensor2, this->input[i]->stride(), this->input[i]->padding(),
                this->input[i]->dilation(), this->input[i]->groups(), &aitisa_result);
    gettimeofday(&aitisa_end,NULL); 
    aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0 
                + (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0;
    aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims, 
                   &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);
    // user
    UserDataType user_dtype1 = UserFuncs::user_int_to_dtype(this->input[i]->dtype1());
    UserDataType user_dtype2 = UserFuncs::user_int_to_dtype(this->input[i]->dtype2());
    UserDevice user_device1 = UserFuncs::user_int_to_device(this->input[i]->device1());
    UserDevice user_device2 = UserFuncs::user_int_to_device(this->input[i]->device2());
    UserFuncs::user_create(user_dtype1, user_device1, this->input[i]->dims1(), 
                           this->input[i]->ndim1(), this->input[i]->data1(),
                           this->input[i]->len1(), &user_tensor1);
    UserFuncs::user_create(user_dtype2, user_device2, this->input[i]->dims2(), 
                           this->input[i]->ndim2(), this->input[i]->data2(), 
                           this->input[i]->len2(), &user_tensor2);
    gettimeofday(&user_start,NULL); 
    UserFuncs::user_conv(user_tensor1, user_tensor2, this->input[i]->stride(), this->input[i]->padding(),
                this->input[i]->dilation(), this->input[i]->groups(), &user_result);
    gettimeofday(&user_end,NULL); 
    user_time = (user_end.tv_sec - user_start.tv_sec) * 1000.0 
              + (user_end.tv_usec - user_start.tv_usec) / 1000.0;
    UserFuncs::user_resolve(user_result, &user_result_dtype, &user_result_device, 
                            &user_result_dims, &user_result_ndim, 
                            (void**)&user_result_data, &user_result_len);
    // compare
    int64_t tensor_size = 1;
    ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
    ASSERT_EQ(/*CUDA*/1, UserFuncs::user_device_to_int(user_result_device));
    ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype), 
              UserFuncs::user_dtype_to_int(user_result_dtype));
    for(int64_t j=0; j<aitisa_result_ndim; j++){
      tensor_size *= aitisa_result_dims[j];
      ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
    }
    ASSERT_EQ(aitisa_result_len, user_result_len);
    float *aitisa_data = (float*)aitisa_result_data;
    float *user_data = (float*)user_result_data;
    for(int64_t j=0; j<tensor_size; j++){
      ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
      // ASSERT_FLOAT_EQ(aitisa_data[j], user_data[j]);
    }
    // print result of test
    std::cout<< /*GREEN <<*/ "[ Conv sample"<< i << " / " 
             << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
    std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
    std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
  }
}
REGISTER_TYPED_TEST_CASE_P(ConvTest, TwoTests);

#define REGISTER_CONV(CONV)                                                               \
  class Conv : public Basic {                                                             \
  public:                                                                                 \
    static void user_conv(UserTensor input, UserTensor filter, const int *stride,         \
                          const int *padding, const int *dilation, const int groups,      \
                          UserTensor *output){                                            \
      CONV(input, filter, stride, padding, dilation, groups, output);                     \
    }                                                                                     \
  };                                                                                      \
  namespace aitisa_api{                                                                   \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, ConvTest, Conv);                            \
  }

} // namespace aitisa_api