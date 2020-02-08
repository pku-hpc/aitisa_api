#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/math/sqrt.h"
//#include "src/tool/tool.h"
}

void sqrt_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 4;
    data[i] = value;
  }
}

void sqrt_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 2;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Sqrt, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  sqrt_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_sqrt(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.000000, 1.414214, 2.000000,
                       2.449490, 2.828427, 3.162278};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Sqrt, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  sqrt_assign_int32(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_sqrt(input, &output);
  // tensor_printer2d(output);

  int32_t* out_data = (int32_t*)aitisa_tensor_data(output);
  int32_t test_data[] = {0, 2, 2, 3, 4, 4};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_EQ(out_data[i], test_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api