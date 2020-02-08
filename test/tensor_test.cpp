#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
}

namespace aitisa_api {
namespace {

TEST(TensorStruct, MemberAccess) {
  Tensor tensor;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {2, 3, 4};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &tensor);
  EXPECT_EQ(3, aitisa_tensor_ndim(tensor));
  EXPECT_EQ(24, aitisa_tensor_size(tensor));
  aitisa_destroy(&tensor);
}

TEST(TensorStruct, Factories) {
  Tensor tensor;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {2, 3, 4};
  aitisa_full(dtype, device, dims, 3, 7.3, &tensor);
  int32_t *data = (int32_t *)aitisa_tensor_data(tensor);
  int64_t size = aitisa_tensor_size(tensor);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(7, data[i]);
  }
  aitisa_destroy(&tensor);

  Tensor tensor2;
  DataType dtype2 = {TYPE_FLOAT, sizeof(float)};
  Device device2 = {DEVICE_CPU, 0};
  int64_t dims2[4] = {1, 2, 3, 4};
  aitisa_full(dtype2, device2, dims2, 4, 2.0, &tensor2);
  float *data2 = (float *)aitisa_tensor_data(tensor2);
  size = aitisa_tensor_size(tensor2);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(2.0, data2[i]);
  }

  aitisa_destroy(&tensor2);
}

}
}