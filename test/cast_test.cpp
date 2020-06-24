#include <math.h>
#include "gtest/gtest.h"
extern "C" {
#include "src/basic/cast.h"
#include "src/basic/factories.h"
#include "src/core/tensor.h"
#include "src/math/binary_op.h"
}

void cast_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value = i;
    data[i] = value;
  }
}

void cast_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {
TEST(Cast, FloatToInt32) {
  Tensor input;
  Tensor temp;
  Tensor factor;
  int64_t dims[2] = {3, 2};
  Device device = {DEVICE_CPU, 0};
  DataType dtype = kFloat;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  aitisa_full(dtype, device, dims, 2, 7, &factor);
  cast_assign_float(input);
  aitisa_mul(input, factor, &temp);
  DataType out_dtype = {TYPE_INT32, sizeof(int32_t)};

  Tensor output;
  Status status;
  status = aitisa_cast(temp, out_dtype, &output);

  DataType in_dtype_test = aitisa_tensor_data_type(input);
  DataType out_dtype_test = aitisa_tensor_data_type(output);
  EXPECT_EQ(in_dtype_test.code, TYPE_FLOAT);
  EXPECT_EQ(out_dtype_test.code, TYPE_INT32);
  int32_t* out_data = (int32_t*)aitisa_tensor_data(output);
  int32_t test_data[] = {0, 0, 1, 2, 2, 3};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_TRUE(out_data[i] == test_data[i]);
  }

  aitisa_destroy(&factor);
  aitisa_destroy(&input);
  aitisa_destroy(&temp);
  aitisa_destroy(&output);
}

TEST(Cast, Int32ToDouble) {
  Tensor input;
  Tensor factor;
  Tensor temp;
  int64_t dims[2] = {3, 2};
  Device device = {DEVICE_CPU, 0};
  DataType dtype = kInt32;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  aitisa_full(dtype, device, dims, 2, 2, &factor);
  cast_assign_int32(input);
  aitisa_mul(input, factor, &temp);
  DataType out_dtype = {TYPE_DOUBLE, sizeof(double)};

  Tensor output;
  Status status;
  status = aitisa_cast(temp, out_dtype, &output);

  DataType in_dtype_test = aitisa_tensor_data_type(input);
  DataType out_dtype_test = aitisa_tensor_data_type(output);
  EXPECT_EQ(in_dtype_test.code, TYPE_INT32);
  EXPECT_EQ(out_dtype_test.code, TYPE_DOUBLE);
  double* out_data = (double*)aitisa_tensor_data(output);
  double test_data[] = {0, 2, 4, 6, 8, 10};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&factor);
  aitisa_destroy(&input);
  aitisa_destroy(&temp);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api