#pragma once

// #include <ctime>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/nn/relu.h"
#include "src/nn/sigmoid.h"
#include "src/nn/tanh.h"
#include <math.h>
#include <sys/time.h>
}

namespace aitisa_api {

template <typename InterfaceType>
class ActivationTest : public ::testing::Test{
public:
  ActivationTest():
    input0(/*ndim*/5, /*dims*/{3,6,10,120,600}, /*dtype=float*/9,  
           /*device1=cuda*/1, /*data*/nullptr, /*len*/0),
    input1(/*ndim*/4, /*dims*/{3,40,100,600}, /*dtype=double*/8,  
           /*device1=cuda*/1, /*data*/nullptr, /*len*/0),
    input2(/*ndim*/4, /*dims*/{3,4,120,60}, /*dtype=float*/8,  
           /*device1=cuda*/1, /*data*/nullptr, /*len*/0){
    input[0] = &input0;
    input[1] = &input1;
    input[2] = &input2;
    ninput = 3;
    for(int i=0; i<ninput; i++){
      unsigned int input_nelem = 1;
      for(unsigned int j=0; j<input[i]->ndim(); j++){
        input_nelem *= input[i]->dims()[j];
      }
      unsigned int input_len = input_nelem * elem_size(input[i]->dtype());
      void *input_data = (void*) new char[input_len];
      random_assign(input_data, input_len, input[i]->dtype());
      input[i]->set_data(input_data, input_len);
    }
  }
  virtual ~ActivationTest(){}
  using InputType = Unary_Input;
  using UserInterface = InterfaceType;
  // inputs
  Unary_Input input0;
  Unary_Input input1;
  Unary_Input input2;
  Unary_Input *input[3] = {&input0, &input1, &input2};
  std::string input0_name = "Random Double CUDA with Dims{3,6,10,120,600} for ReLU";
  std::string input1_name = "Random Float CUDA with Dims{30,40,100,600} for Sigmoid";
  std::string input2_name = "Natural FLoat CUDA with Dims{3,4,120,60} for Tanh";
  std::string *input_name[3] = {&input0_name, &input1_name, &input2_name};
  int ninput = 3;
};
TYPED_TEST_CASE_P(ActivationTest);

TYPED_TEST_P(ActivationTest, ThreeTests){
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;
  for(int i=0; i<this->ninput; i++){
    // if(i != 2) continue;
    struct timeval aitisa_start, aitisa_end, user_start, user_end;
    double aitisa_time, user_time;
    int64_t aitisa_result_ndim, user_result_ndim;
    int64_t *aitisa_result_dims=nullptr, *user_result_dims=nullptr;
    float *aitisa_result_data=nullptr, *user_result_data=nullptr;
    unsigned int aitisa_result_len, user_result_len;
    AITISA_Tensor aitisa_tensor, aitisa_result;
    AITISA_DataType aitisa_result_dtype;
    AITISA_Device aitisa_result_device;
    UserTensor user_tensor, user_result;
    UserDataType user_result_dtype;
    UserDevice user_result_device;
    // aitisa
    AITISA_DataType aitisa_dtype = aitisa_int_to_dtype(this->input[i]->dtype());
    AITISA_Device aitisa_device = aitisa_int_to_device(0); // cpu supoorted only
    aitisa_create(aitisa_dtype, aitisa_device, this->input[i]->dims(), this->input[i]->ndim(), 
                  (void*)(this->input[i]->data()), this->input[i]->len(), &aitisa_tensor);
    gettimeofday(&aitisa_start,NULL); 
    switch(i){
      case 0: aitisa_relu(aitisa_tensor, &aitisa_result); break;
      case 1: aitisa_sigmoid(aitisa_tensor, &aitisa_result); break;
      case 2: aitisa_tanh(aitisa_tensor, &aitisa_result); break;
      default: break;
    }
    gettimeofday(&aitisa_end,NULL); 
    aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0 
                + (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0 ;
    aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims, 
                   &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);
    // user
    UserDataType user_dtype = UserFuncs::user_int_to_dtype(this->input[i]->dtype());
    UserDevice user_device = UserFuncs::user_int_to_device(this->input[i]->device());
    UserFuncs::user_create(user_dtype, user_device, this->input[i]->dims(), 
                           this->input[i]->ndim(), this->input[i]->data(),
                           this->input[i]->len(), &user_tensor);
    gettimeofday(&user_start,NULL); 
    switch(i){
      case 0: UserFuncs::user_relu(user_tensor, &user_result); break;
      case 1: UserFuncs::user_sigmoid(user_tensor, &user_result); break;
      case 2: UserFuncs::user_tanh(user_tensor, &user_result); break;
      default: break;
    }
    gettimeofday(&user_end,NULL); 
    user_time = (user_end.tv_sec - user_start.tv_sec) * 1000.0 
              + (user_end.tv_usec - user_start.tv_usec) / 1000.0;
    UserFuncs::user_resolve(user_result, &user_result_dtype, &user_result_device, 
                            &user_result_dims, &user_result_ndim, 
                            (void**)&user_result_data, &user_result_len);
    // compare
    int64_t tensor_size = 1;
    ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
    ASSERT_EQ( 
        /*CUDA*/1, UserFuncs::user_device_to_int(user_result_device));
    ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype), 
              UserFuncs::user_dtype_to_int(user_result_dtype));
    for(int64_t j=0; j<aitisa_result_ndim; j++){
      tensor_size *= aitisa_result_dims[j];
      ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
    }
    ASSERT_EQ(aitisa_result_len, user_result_len);
    switch(i){
      case 0: {
        double *aitisa_data = (double*)aitisa_result_data;
        double *user_data = (double*)user_result_data;
        for(int64_t j=0; j<tensor_size; j++){
          ASSERT_FLOAT_EQ(aitisa_data[j], user_data[j]);
        }
        break;
      }
      default: {
        float *aitisa_data = (float*)aitisa_result_data;
        float *user_data = (float*)user_result_data;
        for(int64_t j=0; j<tensor_size; j++){
          ASSERT_FLOAT_EQ(aitisa_data[j], user_data[j]);
        }
        break;
      }
    }
    // print result of test
    std::cout<< /*GREEN <<*/ "[ Activation sample"<< i << " / " 
             << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
    std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
    std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
  }
}
REGISTER_TYPED_TEST_CASE_P(ActivationTest, ThreeTests);

#define REGISTER_ACTIVATION(RELU, SIGMOID, TANH)                                    \
  class Activation : public Basic {                                                 \
  public:                                                                           \
    static void user_relu(UserTensor tensor, UserTensor* result){                   \
      RELU(tensor, result);                                                         \
    }                                                                               \
    static void user_sigmoid(UserTensor tensor, UserTensor* result){                \
      SIGMOID(tensor, result);                                                      \
    }                                                                               \
    static void user_tanh(UserTensor tensor, UserTensor* result){                   \
      TANH(tensor, result);                                                         \
    }                                                                               \
  };                                                                                \
  namespace aitisa_api{                                                             \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, ActivationTest, Activation);          \
  }

} // namespace aitisa_api
