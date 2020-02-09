#include "gtest/gtest.h"
#include "auto_testing/basic.h"
extern "C" {
#include "src/math/binary_op.h"
}
template <typename FuncType>
class AddTest : public testing::Test{
public:
  AddTest():dtype1(8),dtype2(8),device1(0),device2(0),
            layout_type1(0),layout_type2(0),data1(NULL),data2(NULL){
    ndim1 = ndim2 = 3;
    dims1 = new int64_t[ndim1];
    for(int64_t i=0; i<ndim1; i++){
      dims1[i] = i+1;
    }
    dims2 = dims1;
  }
  ~AddTest(){
    delete [] dims1;
  }
  using UserFuncs = FuncType;
  int64_t ndim1;
  int64_t *dims1;
  int dtype1;
  int device1;
  int layout_type1;
  void *data1;

  int64_t ndim2;
  int64_t *dims2;
  int dtype2;
  int device2;
  int layout_type2;
  void *data2;
};
TYPED_TEST_CASE_P(AddTest);
 
TYPED_TEST_P(AddTest, RandUniformFloat){
  using AITISA_Tensor = Tensor;
  // using UserDataType = typename TestFixture::UserFuncs::UserDataType;
  // using UserDevice = typename TestFixture::UserFuncs::UserDevice;
  // using UserLayoutType = typename TestFixture::UserFuncs::UserLayoutType;
  using UserTensor = typename TestFixture::UserFuncs::UserTensor;
  using UserFuncs = typename TestFixture::UserFuncs;
  UserTensor user_tensor1, user_tensor2, user_result;
  AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
  int64_t aitisa_result_ndim, user_result_ndim;
  int64_t *aitisa_result_dims=nullptr, *user_result_dims=nullptr;
  int aitisa_result_dtype, user_result_dtype;
  float *aitisa_result_data=nullptr, *user_result_data=nullptr;
  // aitisa
  TRANSLATE_DATA_TYPE(this->dtype1, dtype1);
  TRANSLATE_DATA_TYPE(this->dtype2, dtype2);
  TRANSLATE_DEVICE_TYPE(this->device1, device1);
  TRANSLATE_DEVICE_TYPE(this->device2, device2);
  TRANSLATE_LAYOUT_TYPE(this->layout_type1, layout_type1);
  TRANSLATE_LAYOUT_TYPE(this->layout_type2, layout_type2);
  aitisa_create(dtype1, device1, layout_type1, this->dims1, this->ndim1, this->data1, &aitisa_tensor1);
  aitisa_create(dtype2, device2, layout_type2, this->dims2, this->ndim2, this->data2, &aitisa_tensor2);
  aitisa_add(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
  // aitisa_resolve(aitisa_result, &aitisa_result_ndim, aitisa_result_dims, 
  //                &aitisa_result_dtype, &aitisa_result_data);
  // user
  UserFuncs::create(this->dtype1, this->device1, this->layout_type1, this->dims1, this->ndim1, this->data1, &user_tensor1);
  UserFuncs::create(this->dtype1, this->device1, this->layout_type1, this->dims1, this->ndim1, this->data1, &user_tensor1);
  UserFuncs::add(user_tensor1, user_tensor2, &user_result);
  UserFuncs::resolve(user_result, &user_result_dtype, &user_result_ndim, user_result_dims, user_result_data);
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
REGISTER_TYPED_TEST_CASE_P(AddTest, RandUniformFloat);

#define REGISTER_ADD(ADD_FUNC)                                                      \
  class Add : public TensorBase {                                                   \
  public:                                                                           \
    static void add(UserTensor tensor1, UserTensor tensor2, UserTensor* result){    \
      ADD_FUNC(tensor1, tensor2, result);                                           \
    }                                                                               \
  };                                                                                \
  INSTANTIATE_TYPED_TEST_CASE_P(RandUniformFloat, AddTest, Add);
  