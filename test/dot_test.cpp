#include <math.h>
#include <vector>
#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/core/tensor.h"
#include "src/math/dot.h"
//#include "src/tool/tool.h"
}

void dot_assign_float(Tensor t) {
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

TEST(Dot, Dim2DotDim2) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};

  int64_t dims[2] = {3, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 2, NULL, 0, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  // tensor_printer2d(tensor1);
  // tensor_printer2d(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.15, 0.18, 0.21, 0.42, 0.54, 0.66, 0.69, 0.90, 1.11};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Dot, Dim2DotDim0) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};

  int64_t dims1[2] = {3, 3};
  int64_t* dims2 = NULL;
  aitisa_create(dtype, device, dims1, 2, NULL, 0, &tensor1);
  aitisa_full(dtype, device, dims2, 0, 2, &tensor2);
  dot_assign_float(tensor1);
  // tensor_printer2d(tensor1);
  // tensor_printer(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.00, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Dot, Dim1DotDim1) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};

  int64_t dims[1] = {5};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  // tensor_printer(tensor1);
  // tensor_printer(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  // tensor_printer(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.30};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Dot, Dim3DotDim1) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};

  int64_t dims1[3] = {2, 5, 6};
  int64_t dims2[1] = {6};
  aitisa_create(dtype, device, dims1, 3, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims2, 1, NULL, 0, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  // printf("tensor1:\n");
  // tensor_printer2d(tensor1);
  // printf("tensor2:\n");
  // tensor_printer(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.55, 1.45, 2.35, 3.25, 4.15,
                       5.05, 5.95, 6.85, 7.75, 8.65};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Dot, Dim4DotDim3) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};

  int64_t dims1[4] = {2, 3, 2, 3};
  int64_t dims2[3] = {2, 3, 2};
  aitisa_create(dtype, device, dims1, 4, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims2, 3, NULL, 0, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  // tensor_printer2d(tensor1);
  // tensor_printer2d(tensor2);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {
      0.10, 0.13, 0.28, 0.31, 0.28, 0.40, 1.00, 1.12, 0.46, 0.67, 1.72, 1.93,
      0.64, 0.94, 2.44, 2.74, 0.82, 1.21, 3.16, 3.55, 1.00, 1.48, 3.88, 4.36,
      1.18, 1.75, 4.60, 5.17, 1.36, 2.02, 5.32, 5.98, 1.54, 2.29, 6.04, 6.79,
      1.72, 2.56, 6.76, 7.60, 1.90, 2.83, 7.48, 8.41, 2.08, 3.10, 8.20, 9.22};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api