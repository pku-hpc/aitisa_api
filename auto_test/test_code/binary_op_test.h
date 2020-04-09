#pragma once

#include <ctime>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/math/binary_op.h"
#include <math.h>
}

namespace aitisa_api {

template <typename InterfaceType>
class BinaryOPTest : public ::testing::Test{
public:
  BinaryOPTest():
    input0(/*ndim1*/2, /*dims1*/{10,6}, /*dtype1=int32*/4,  
           /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
           /*ndim2*/2, /*dims2*/{10,6}, /*dtype2=float*/4, 
           /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input1(/*ndim1*/2, /*dims1*/{2013,2018}, /*dtype1=double*/9,  
           /*device1=cuda*/1, /*data1*/nullptr, /*len1*/0, 
           /*ndim2*/2, /*dims2*/{2013,2018}, /*dtype2=double*/9, 
           /*device2=cuda*/1, /*data2*/nullptr, /*len2*/0),
    input2(/*ndim1*/3, /*dims1*/{10,3,2}, /*dtype1=uint64*/7,  
           /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
           /*ndim2*/3, /*dims2*/{10,3,2}, /*dtype2=uint64*/7, 
           /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input3(/*ndim1*/1, /*dims1*/{5}, /*dtype1=float*/8,  
           /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
           /*ndim2*/1, /*dims2*/{5}, /*dtype2=float*/8, 
           /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0){
    input[0] = &input0;
    input[1] = &input1;
    input[2] = &input2;
    input[3] = &input3;
    ninput = 4;
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
      if(i == 1){ 
        random_assign(input_data1, input_len1, input[i]->dtype1());
        random_assign(input_data2, input_len2, input[i]->dtype2());
      }else{
        natural_assign(input_data1, input_len1, input[i]->dtype1());
        natural_assign(input_data2, input_len2, input[i]->dtype2());
      }
      input[i]->set_data1(input_data1, input_len1);
      input[i]->set_data2(input_data2, input_len2);
    }
  }
  virtual ~BinaryOPTest(){}
  using InputType = Binary_Input;
  using UserInterface = InterfaceType;
  // inputs
  Binary_Input input0; // Natural assigned int32 type input of CPU with dims1{10,6} and dims2{10,6} for add
  Binary_Input input1; // Random assigned double type input of CUDA with dims1{2013,2018} and dims2{2013,2018} for sub
  Binary_Input input2; // Natural assigned uint64 type input of CPU with dims1{10,3,2} and dims2{10,3,2} for mul
  Binary_Input input3; // Natural assigned float type input of CPU with dims1{5} and dims2{5} for div
  Binary_Input *input[4] = {&input0, &input1, &input2, &input3};
  std::string input0_name = "Natural int32 CPU with Dims{10,6} and Dims{10,6} for add";
  std::string input1_name = "Random Double CUDA with Dims{2013,2018} and Dims{2013,2018} dor sub";
  std::string input2_name = "Natural uint64 CPU with Dims{10,3,2} and Dims{10,3,2} for mul";
  std::string input3_name = "Natural Float CPU with Dims{5} and Dims{5} for div";
  std::string *input_name[4] = {&input0_name, &input1_name, &input2_name, &input3_name};
  int ninput = 4;
};
TYPED_TEST_CASE_P(BinaryOPTest);

TYPED_TEST_P(BinaryOPTest, FourTests){
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;
  for(int i=0; i<this->ninput; i++){
    // if(i==1) continue;
    std::clock_t aitisa_start, aitisa_end, user_start, user_end;
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
    aitisa_start = std::clock();
    switch(i){
      case 0: aitisa_add(aitisa_tensor1, aitisa_tensor2, &aitisa_result); break;
      case 1: aitisa_sub(aitisa_tensor1, aitisa_tensor2, &aitisa_result); break;
      case 2: aitisa_mul(aitisa_tensor1, aitisa_tensor2, &aitisa_result); break;
      case 3: aitisa_div(aitisa_tensor1, aitisa_tensor2, &aitisa_result); break;
      default: break;
    }
    aitisa_end = std::clock();
    aitisa_time = (double)(aitisa_end - aitisa_start) / CLOCKS_PER_SEC * 1000;
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
    user_start = std::clock();
    switch(i){
      case 0: UserFuncs::user_add(user_tensor1, user_tensor2, &user_result); break;
      case 1: UserFuncs::user_sub(user_tensor1, user_tensor2, &user_result); break;
      case 2: UserFuncs::user_mul(user_tensor1, user_tensor2, &user_result); break;
      case 3: UserFuncs::user_div(user_tensor1, user_tensor2, &user_result); break;
      default: break;
    }
    user_end = std::clock();
    user_time = (double)(user_end - user_start) / CLOCKS_PER_SEC * 1000;
    UserFuncs::user_resolve(user_result, &user_result_dtype, &user_result_device, 
                            &user_result_dims, &user_result_ndim, 
                            (void**)&user_result_data, &user_result_len);
    // compare
    int64_t tensor_size = 1;
    ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
    if(i == 1){ // CUDA
      ASSERT_EQ(
        /*CUDA*/1, UserFuncs::user_device_to_int(user_result_device));
    }else{ // CPU
      ASSERT_EQ(aitisa_device_to_int(aitisa_result_device), 
                UserFuncs::user_device_to_int(user_result_device));
    }
    ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype), 
              UserFuncs::user_dtype_to_int(user_result_dtype));
    for(int64_t j=0; j<aitisa_result_ndim; j++){
      tensor_size *= aitisa_result_dims[j];
      ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
    }
    ASSERT_EQ(aitisa_result_len, user_result_len);
    switch(i){
      case 0: {
        int32_t *aitisa_data = (int32_t*)aitisa_result_data;
        int32_t *user_data = (int32_t*)user_result_data;
        for(int64_t j=0; j<tensor_size; j++){
          ASSERT_EQ(aitisa_data[j], user_data[j]);
        }
        break;
      }
      case 1: {
        double *aitisa_data = (double*)aitisa_result_data;
        double *user_data = (double*)user_result_data;
        for(int64_t j=0; j<tensor_size; j++){
          ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
        }
        break;
      }
      case 2: {
        uint64_t *aitisa_data = (uint64_t*)aitisa_result_data;
        uint64_t *user_data = (uint64_t*)user_result_data;
        for(int64_t j=0; j<tensor_size; j++){
          ASSERT_EQ(aitisa_data[j], user_data[j]);
        }
        break;
      }
      case 3: {
        float *aitisa_data = (float*)aitisa_result_data;
        float *user_data = (float*)user_result_data;
        for(int64_t j=0; j<tensor_size; j++){
          ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
        }
        break;
      }
      default: break;
    }
    // print result of test
    std::cout<< /*GREEN <<*/ "[ BinaryOP sample"<< i << " / " 
             << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
    std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
    std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
  }
}
REGISTER_TYPED_TEST_CASE_P(BinaryOPTest, FourTests);

#define REGISTER_BINARY_OP(ADD, SUB, MUL, DIV)                                            \
  class BinaryOP : public Basic {                                                         \
  public:                                                                                 \
    static void user_add(UserTensor tensor1, UserTensor tensor2, UserTensor* result){     \
      ADD(tensor1, tensor2, result);                                                      \
    }                                                                                     \
    static void user_sub(UserTensor tensor1, UserTensor tensor2, UserTensor* result){     \
      SUB(tensor1, tensor2, result);                                                      \
    }                                                                                     \
    static void user_mul(UserTensor tensor1, UserTensor tensor2, UserTensor* result){     \
      MUL(tensor1, tensor2, result);                                                      \
    }                                                                                     \
    static void user_div(UserTensor tensor1, UserTensor tensor2, UserTensor* result){     \
      DIV(tensor1, tensor2, result);                                                      \
    }                                                                                     \
  };                                                                                      \
  namespace aitisa_api{                                                                   \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, BinaryOPTest, BinaryOP);                    \
  }

} // namespace aitisa_api
