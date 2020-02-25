#pragma once

#include <ctime>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/math/matmul.h"
#include <math.h>
}

namespace aitisa_api {

template <typename InterfaceType>
class MatmulTest : public ::testing::Test{
public:
  MatmulTest():
    input0(/*ndim1*/1, /*dims1*/{10}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/1, /*dims2*/{10}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input1(/*ndim1*/2, /*dims1*/{199,202}, /*dtype1=double*/9,  
            /*device1=cuda*/1, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/2, /*dims2*/{202,201}, /*dtype2=double*/9, 
            /*device2=cuda*/1, /*data2*/nullptr, /*len2*/0),
    input2(/*ndim1*/1, /*dims1*/{10}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/2, /*dims2*/{10,5}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input3(/*ndim1*/2, /*dims1*/{10,5}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/1, /*dims2*/{5}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input4(/*ndim1*/1, /*dims1*/{3}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/5, /*dims2*/{2,2,4,3,2}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input5(/*ndim1*/5, /*dims1*/{2,2,4,2,3}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/1, /*dims2*/{3}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input6(/*ndim1*/3, /*dims1*/{2,4,3}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/4, /*dims2*/{3,2,3,2}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0){
    input[0] = &input0;
    input[1] = &input1;
    input[2] = &input2;
    input[3] = &input3;
    input[4] = &input4;
    input[5] = &input5;
    input[6] = &input6;
    ninput = 7;
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
  virtual ~MatmulTest(){}
  using UserInterface = InterfaceType;
  // inputs
  Binary_Input input0; // Natural assigned float type input of CPU with dims1{10} and dims2{10}
  Binary_Input input1; // Random assigned double type input of CUDA with dims1{1995,2020} and dims2{2020,2018}
  Binary_Input input2; // Natural assigned float type input of CPU with dims1{10} and dims2{10,5}
  Binary_Input input3; // Natural assigned float type input of CPU with dims1{10,5} and dims2{5}
  Binary_Input input4; // Natural assigned float type input of CPU with dims1{3} and dims2{2,2,4,3,2}
  Binary_Input input5; // Natural assigned float type input of CPU with dims1{2,2,4,2,3} and dims2{3}
  Binary_Input input6; // Natural assigned float type input of CPU with dims1{2,4,3} and dims2{3,2,3,2}
  Binary_Input *input[7] = {&input0, &input1, &input2, &input3, &input4, &input5, &input6};
  std::string input0_name = "Natural Float CPU with Dims{10} and Dims{10}";
  std::string input1_name = "Random Double CUDA with Dims{199,202} and Dims{202,201}";
  std::string input2_name = "Natural Float CPU with Dims{10} and Dims{10,5}";
  std::string input3_name = "Natural Float CPU with Dims{10,5} and Dims{5}";
  std::string input4_name = "Natural Float CPU with Dims{3} and Dims{2,2,4,3,2}";
  std::string input5_name = "Natural Float CPU with Dims{2,2,4,2,3} and Dims{3}";
  std::string input6_name = "Natural Float CPU with Dims{2,4,3} and Dims{3,2,3,2}";
  std::string *input_name[7] = {&input0_name, &input1_name, &input2_name, &input3_name, 
                                &input4_name, &input5_name, &input6_name};
  int ninput = 7;
};
TYPED_TEST_CASE_P(MatmulTest);

TYPED_TEST_P(MatmulTest, FiveTests){
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
    aitisa_matmul(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
    aitisa_end = std::clock();
    aitisa_time = (double)(aitisa_end - aitisa_start) / CLOCKS_PER_SEC * 1000;
    aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims, 
                   &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);
    //for debug, please delete it when it is done!
//   // print_data2d((float*)this->input2.data1(), 3, 3);
//   // print_data2d((float*)this->input2.data2(), 3, 3);
//   // print_data2d(aitisa_result_data, 3, 3);
//   // if(aitisa_result_dtype.code == TYPE_FLOAT) std::cout<<"dtype yes!"<<std::endl;
//   // if(aitisa_result_device.type == DEVICE_CPU) std::cout<<"device yes!"<<std::endl;
//   // if(aitisa_result_ndim == 2) std::cout<<"ndim yes!"<<std::endl;
//   // if(aitisa_result_dims[0]==3 && aitisa_result_dims[1]==2) std::cout<<"ndim yes!"<<std::endl;
//   // for(int64_t i=0; i<aitisa_tensor_size(aitisa_result); i++) std::cout<< aitisa_result_data[i] <<std::endl;
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
    UserFuncs::user_matmul(user_tensor1, user_tensor2, &user_result);
    user_end = std::clock();
    user_time = (double)(user_end - user_start) / CLOCKS_PER_SEC * 1000;
    UserFuncs::user_resolve(user_result, &user_result_dtype, &user_result_device, 
                            &user_result_dims, &user_result_ndim, 
                            (void**)&user_result_data, &user_result_len);
    // compare
    int64_t tensor_size = 1;
    EXPECT_EQ(aitisa_result_ndim, user_result_ndim);
    if(i == 1){ // CUDA
      EXPECT_EQ(
        /*CUDA*/1, UserFuncs::user_device_to_int(user_result_device));
    }else{ // CPU
      EXPECT_EQ(aitisa_device_to_int(aitisa_result_device), 
                UserFuncs::user_device_to_int(user_result_device));
    }
    EXPECT_EQ(aitisa_dtype_to_int(aitisa_result_dtype), 
              UserFuncs::user_dtype_to_int(user_result_dtype));
    for(int64_t j=0; j<aitisa_result_ndim; j++){
      tensor_size *= aitisa_result_dims[j];
      EXPECT_EQ(aitisa_result_dims[j], user_result_dims[j]);
    }
    EXPECT_EQ(aitisa_result_len, user_result_len);
    if(i == 1){ // Double
      // std::cout<< "ok1" << std::endl;
      double *aitisa_data = (double*)aitisa_result_data;
      double *user_data = (double*)user_result_data;
      for(int64_t j=0; j<tensor_size; j++){
        EXPECT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
      }
      // std::cout<< "ok2" << std::endl;
    }else{ // Float
      for(int64_t j=0; j<tensor_size; j++){
        EXPECT_TRUE(abs(aitisa_result_data[j] - user_result_data[j]) < 1e-3);
      }
    }
    // print result of test
    std::cout<< /*GREEN <<*/ "[ Matmul sample"<< i << " / " 
             << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
    std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
    std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
  }
}
REGISTER_TYPED_TEST_CASE_P(MatmulTest, FiveTests);

//  void get_sample_matmul(Sample<Binary_Input>& sample, std::string case_name){
//   std::string case1("aitisa_api/MatmulTest/0.NaturalFloatCPU");
//   std::string case2("aitisa_api/MatmulTest/0.RandomDoubleCUDA");
//   int sample_num = 0;
//   AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
//   AITISA_DataType aitisa_result_dtype;
//   AITISA_Device aitisa_result_device;
//   int64_t aitisa_result_ndim;
//   int64_t *aitisa_result_dims = nullptr;
//   void *aitisa_result_data=nullptr;
//   unsigned int aitisa_result_len;
//   Concrete<MatmulTest<void>> matmul_test;
//   if(case_name == case1) {
//     sample_num = 1;
//   }else if(case_name == case2){
//     sample_num = 2;
//   }else {}
//   switch(sample_num){
//     case 1: sample.set_input(matmul_test.input1); break;
//     case 2: sample.set_input(matmul_test.input2); break;
//     default: break;
//   }
//   AITISA_DataType aitisa_dtype1 = aitisa_int_to_dtype(sample.input().dtype1());
//   AITISA_DataType aitisa_dtype2 = aitisa_int_to_dtype(sample.input().dtype2());
//   AITISA_Device aitisa_device1 = aitisa_int_to_device(0); // cpu supported only
//   AITISA_Device aitisa_device2 = aitisa_int_to_device(0); // cpu supported only
//   aitisa_create(aitisa_dtype1, aitisa_device1, sample.input().dims1(), sample.input().ndim1(), 
//                 (void*)(sample.input().data1()), sample.input().len1(), &aitisa_tensor1);
//   aitisa_create(aitisa_dtype2, aitisa_device2, sample.input().dims2(), sample.input().ndim2(), 
//                 (void*)(sample.input().data2()), sample.input().len2(), &aitisa_tensor2);
//   aitisa_matmul(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
//   aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims, 
//                  &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);
//   sample.set_result(aitisa_result_ndim, aitisa_result_dims, aitisa_result_data, aitisa_result_len);
// }

#define REGISTER_MATMUL(MATMUL_FUNC)                                                      \
  class Matmul : public Basic {                                                           \
  public:                                                                                 \
    static void user_matmul(UserTensor tensor1, UserTensor tensor2, UserTensor* result){  \
      MATMUL_FUNC(tensor1, tensor2, result);                                              \
    }                                                                                     \
  };                                                                                      \
  namespace aitisa_api{                                                                   \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, MatmulTest, Matmul);                        \
  }
  
} // namespace aitisa_api