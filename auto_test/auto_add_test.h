#include "gtest/gtest.h"
#include "auto_test/basic.h"
extern "C" {
#include "src/math/binary_op.h"
}

template <typename FuncType>
class AddTest : public testing::Test{
public:
  AddTest():
    nat_float_sample(/*ndim1*/3, /*ndim2*/3, /*dims1*/{2,3,4}, /*dims2*/{2,3,4}, 
                     /*dtype1=float*/8, /*dtype2=float*/8, 
                     /*device1=cpu*/0, /*device2=cpu*/0, 
                     /*data1*/NULL, /*data2*/NULL, /*len1*/0, /*len2*/0){
    // natural assigned float sample
    int64_t nat_float_sample_nelem = 1;
    for(int64_t i=0; i<nat_float_sample.ndim1_; i++){
      nat_float_sample_nelem *= nat_float_sample.dims1_[i];
    }
    nat_float_sample.len1_ = nat_float_sample_nelem * sizeof(float);
    nat_float_sample.len2_ = nat_float_sample.len1_;
    nat_float_sample.data1_ = new float[nat_float_sample_nelem];
    nat_float_sample.data2_ = new float[nat_float_sample_nelem];
    natural_assign_float(nat_float_sample.data1_, nat_float_sample_nelem);
    natural_assign_float(nat_float_sample.data2_, nat_float_sample_nelem);
  }
  ~AddTest(){}
  using UserFuncs = FuncType;
  // Samples
  Binary_Sample<float> nat_float_sample;
};
TYPED_TEST_CASE_P(AddTest);
 
TYPED_TEST_P(AddTest, NaturalFloat){
  using AITISA_Tensor = Tensor;
  using AITISA_Device = Device;
  using AITISA_DataType = DataType;
  using UserDataType = typename TestFixture::UserFuncs::UserDataType;
  using UserDevice = typename TestFixture::UserFuncs::UserDeviceType;
  using UserTensor = typename TestFixture::UserFuncs::UserTensor;
  using UserFuncs = typename TestFixture::UserFuncs;
  UserTensor user_tensor1, user_tensor2, user_result;
  AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
  int64_t aitisa_result_ndim, user_result_ndim;
  int64_t *aitisa_result_dims=nullptr, *user_result_dims=nullptr;
  AITISA_DataType aitisa_result_dtype;
  int user_result_dtype;
  AITISA_Device aitisa_result_device;
  int user_result_device;
  float *aitisa_result_data=nullptr, *user_result_data=nullptr;
  int64_t aitisa_result_len, user_result_len;
  // aitisa
  TRANSLATE_DATA_TYPE(this->nat_float_sample.dtype1_, dtype1);
  TRANSLATE_DATA_TYPE(this->nat_float_sample.dtype2_, dtype2);
  TRANSLATE_DEVICE_TYPE(this->nat_float_sample.device1_, device1);
  TRANSLATE_DEVICE_TYPE(this->nat_float_sample.device2_, device2);
  aitisa_create(dtype1, device1, this->nat_float_sample.dims1_, this->nat_float_sample.ndim1_, 
                (void*)(this->nat_float_sample.data1_), this->nat_float_sample.len1_, &aitisa_tensor1);
  aitisa_create(dtype2, device2, this->nat_float_sample.dims2_, this->nat_float_sample.ndim2_, 
                (void*)(this->nat_float_sample.data2_), this->nat_float_sample.len2_, &aitisa_tensor2);
  aitisa_add(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
  aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims, &aitisa_result_ndim,  
                 (void**)&aitisa_result_data, &aitisa_result_len);
  //for debug, please delete it when it is done!
  // if(aitisa_result_dtype.code == TYPE_FLOAT) std::cout<<"dtype yes!"<<std::endl;
  // if(aitisa_result_device.type == DEVICE_CPU) std::cout<<"device yes!"<<std::endl;
  // if(aitisa_result_ndim == 3) std::cout<<"ndim yes!"<<std::endl;
  // if(aitisa_result_dims[0]==2 && aitisa_result_dims[1]==3 && aitisa_result_dims[2]==4) std::cout<<"ndim yes!"<<std::endl;
  // for(int64_t i=0; i<aitisa_tensor_size(aitisa_result); i++) std::cout<< aitisa_result_data[i] <<std::endl;

  // user
  UserFuncs::create(this->nat_float_sample.dtype1_, this->nat_float_sample.device1_, this->nat_float_sample.dims1_, 
                    this->nat_float_sample.ndim1_, this->nat_float_sample.data1_, this->nat_float_sample.len1_, &user_tensor1);
  UserFuncs::create(this->nat_float_sample.dtype2_, this->nat_float_sample.device2_, this->nat_float_sample.dims2_, 
                    this->nat_float_sample.ndim2_, this->nat_float_sample.data2_, this->nat_float_sample.len2_, &user_tensor2);
  UserFuncs::add(user_tensor1, user_tensor2, &user_result);
  UserFuncs::resolve(user_result, &user_result_dtype, &user_result_device, &user_result_dims, 
                     &user_result_ndim, (void**)&user_result_data, &user_result_len);
  // compare
  // int64_t size = 1;
  // EXPECT_EQ(aitisa_result_ndim, user_result_ndim);
  // for(int64_t i=0; i<aitisa_result_ndim; i++){
  //   size *= aitisa_result_dims[i];
  //   EXPECT_EQ(aitisa_result_dims[i], user_result_dims[i]);
  // }
  // EXPECT_EQ(aitisa_result_dtype, user_result_dtype);
  // for(int64_t i=0; i<size; i++){
  //   EXPECT_FLOAT_EQ(aitisa_result_data[i], user_result_data[i]);
  // }
}
REGISTER_TYPED_TEST_CASE_P(AddTest, NaturalFloat);

#define REGISTER_ADD(ADD_FUNC)                                                      \
  class Add : public TensorBase {                                                   \
  public:                                                                           \
    static void add(UserTensor tensor1, UserTensor tensor2, UserTensor* result){    \
      ADD_FUNC(tensor1, tensor2, result);                                           \
    }                                                                               \
  };                                                                                \
  INSTANTIATE_TYPED_TEST_CASE_P(whatever, AddTest, Add);
