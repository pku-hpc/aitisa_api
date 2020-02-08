#include <math.h>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/core/tensor.h"
#include "src/nn/pooling.h"
//#include "src/tool/tool.h"
}

void pooling_assign_float(Tensor t) {
  int64_t ndim = aitisa_tensor_ndim(t);
  int64_t* dims = aitisa_tensor_dims(t);
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

void pooling_assign_double(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  double* data = (double*)aitisa_tensor_data(t);
  double value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

void pooling_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    data[i] = i;
  }
}

namespace aitisa_api {
namespace {

TEST(Pooling1D, MaxFloatK2S2P0D2) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {2, 3, 9};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  pooling_assign_float(input);

  Tensor output;
  int ksize[1] = {2};
  int stride[1] = {2};
  int padding[1] = {0};
  int dilation[1] = {2};
  aitisa_pooling(input, "max", ksize, stride, padding, dilation, &output);
  //tensor_printer2d(input);
  //tensor_printer2d(output);

  float* output_data = (float*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  float test_data[] = {0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.5, 1.7,
                       2.0, 2.2, 2.4, 2.6, 2.9, 3.1, 3.3, 3.5,
                       3.8, 4.0, 4.2, 4.4, 4.7, 4.9, 5.1, 5.3};

  for (int i = 0; i < output_size; i++) {
    // Due to the problem of precision, consider the two numbers are equal when
    // their difference is less than 0.0001
    EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Pooling2D, AvgInt32K3S3P0D1) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {1, 2, 9, 9};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  pooling_assign_int32(input);

  Tensor output;
  int ksize[2] = {3, 3};
  int stride[2] = {3, 3};
  int padding[2] = {0, 0};
  int dilation[2] = {1, 1};
  aitisa_pooling(input, "avg", ksize, stride, padding, dilation, &output);
  //tensor_printer2d(input);
  //tensor_printer2d(output);

  int32_t* output_data = (int32_t*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  int32_t test_data[] = {10, 13, 16, 37,  40,  43,  64,  67,  70,
                         91, 94, 97, 118, 121, 124, 145, 148, 151};
  for (int i = 0; i < output_size; i++) {
    EXPECT_TRUE(output_data[i] == test_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Pooling2D, MaxDoubleK6S6P3D1) {
  Tensor input;
  DataType dtype = kDouble;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {1, 2, 15, 15};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  pooling_assign_double(input);

  Tensor output;
  int ksize[2] = {6, 6};
  int stride[2] = {6, 6};
  int padding[2] = {3, 3};
  int dilation[2] = {1, 1};
  aitisa_pooling(input, "max", ksize, stride, padding, dilation, &output);
  //tensor_printer2d(input);
  //tensor_printer2d(output);

  double* output_data = (double*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  double test_data[] = { 3.2000,  3.8000,  4.4000, 12.2000, 12.8000, 13.4000,
                        21.2000, 21.8000, 22.4000, 25.7000, 26.3000, 26.9000,
                        34.7000, 35.3000, 35.9000, 43.7000, 44.3000, 44.9000 };

  for (int i = 0; i < output_size; i++) {
    // Due to the problem of precision, consider the two numbers are equal when
    // their difference is less than 0.0001
    EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Pooling3D, MaxInt32K3S5P0D2) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[5] = {1, 2, 5, 10, 9};
  aitisa_create(dtype, device, dims, 5, NULL, 0, &input);
  pooling_assign_int32(input);

  Tensor output;
  int ksize[3] = {3, 3, 3};
  int stride[3] = {5, 5, 5};
  int padding[3] = {0, 0, 1};
  int dilation[3] = {2, 2, 2};
  aitisa_pooling(input, "max", ksize, stride, padding, dilation, &output);
  //tensor_printer2d(input);
  //tensor_printer2d(output);

  int32_t* output_data = (int32_t*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  int32_t test_data[] = {399, 404, 444, 449, 849, 854, 894, 899};

  for (int i = 0; i < output_size; i++) {
    EXPECT_TRUE(output_data[i] == test_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api
