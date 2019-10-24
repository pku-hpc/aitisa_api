#include "gtest/gtest.h"
#include "gmock/gmock.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/dropout.h"
#include <math.h>
}

void dropout_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1 + 1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {
TEST(Dropout, Rate40percent) {
  Tensor input;
  int64_t dims[2] = { 30,20 };
  Device device = { DEVICE_CPU, 0 };
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 2, &input);
  dropout_assign_float(input);
  
  Tensor output;
  double rate = 0.4;
  aitisa_dropout(input, rate, &output);
  
  int64_t expect_nzeros = (int64_t)(rate * aitisa_tensor_size(input));
  float* output_data = (float*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  int64_t test_nzeros = 0;
  for (int64_t i = 0; i < output_size; i++) {
    if (output_data[i] == 0) {
      test_nzeros++;
    }
  }
  double actual_rate = (double)test_nzeros / (double)output_size;
  EXPECT_TRUE(abs(actual_rate-rate)<0.01);
}
}//namespace
}//namespace aitisa_api

