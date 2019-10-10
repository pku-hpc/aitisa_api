#include "gtest/gtest.h"
#include <math.h>
#include <vector> 
extern "C" {
#include "src/math/dot.h"
#include "src/basic/factories.h"
#include "src/core/tensor.h"
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
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  Device device = { DEVICE_CPU, 0 };
  
  int64_t dims[2] = { 3, 3 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 2, &tensor1);
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 2, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  //tensor_printer2d(tensor1);
  //tensor_printer2d(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  //tensor_printer2d(output);
  
  float* out_data = (float*)aitisa_tensor_data(output);
  std::vector<float> test_data = { 0.15, 0.18, 0.21,
										               0.42, 0.54, 0.66,
                                   0.69, 0.90, 1.11};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
	  // Due to the problem of precision, consider the two numbers 
		// are equal when their difference is less than 0.00001
		EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
	}
}

TEST(Dot, Dim2DotDim0) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  Device device = { DEVICE_CPU, 0 };
  
  int64_t dims1[2] = { 3,3 };
  int64_t* dims2 = NULL;
  aitisa_create(dtype, device, LAYOUT_DENSE, dims1, 2, &tensor1);
  aitisa_full(dtype, device, dims2, 0, 2, &tensor2);
  dot_assign_float(tensor1);
  //tensor_printer2d(tensor1);
  //tensor_printer2d(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  //tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  std::vector<float> test_data = { 0.00, 0.20, 0.40,
                                   0.60, 0.80, 1.00,
                                   1.20, 1.40, 1.60 };
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers 
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }
}

TEST(Dot, Dim1DotDim1) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  Device device = { DEVICE_CPU, 0 };

  int64_t dims[1] = { 5 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 1, &tensor1);
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 1, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  //tensor_printer2d(tensor1);
  //tensor_printer2d(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  //tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  std::vector<float> test_data = { 0.30 };
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers 
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }
}

TEST(Dot, Dim3DotDim1) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  Device device = { DEVICE_CPU, 0 };

  int64_t dims1[3] = { 2,5,6 };
  int64_t dims2[1] = { 6 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims1, 3, &tensor1);
  aitisa_create(dtype, device, LAYOUT_DENSE, dims2, 1, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  //tensor_printer2d(tensor1);
  //tensor_printer2d(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  //tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  std::vector<float> test_data = { 0.55, 1.45, 2.35, 3.25, 4.15,
                                   5.05, 5.95, 6.85, 7.75, 8.65 };
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers 
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }
}

TEST(Dot, Dim4DotDim3) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  Device device = { DEVICE_CPU, 0 };

  int64_t dims1[4] = { 2,3,2,3 };
  int64_t dims2[3] = { 3,1,2 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims1, 4, &tensor1);
  aitisa_create(dtype, device, LAYOUT_DENSE, dims2, 3, &tensor2);
  dot_assign_float(tensor1);
  dot_assign_float(tensor2);
  //tensor_printer2d(tensor1);
  //tensor_printer2d(tensor2);

  Tensor output;
  aitisa_dot(tensor1, tensor2, &output);
  //tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  std::vector<float> test_data = { 0.10, 0.13, 0.28, 0.40, 0.46, 0.67,
                                   0.64, 0.94, 0.82, 1.21, 1.00, 1.48,
                                   1.18, 1.75, 1.36, 2.02, 1.54, 2.29,
                                   1.72, 2.56, 1.90, 2.83, 2.08, 3.10};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    // Due to the problem of precision, consider the two numbers 
    // are equal when their difference is less than 0.00001
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }
}

}//namespace
}//namespace aitisa_api