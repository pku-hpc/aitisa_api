#include <math.h>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/dropout.h"
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
  int64_t dims[2] = {30, 20};
  Device device = {DEVICE_CPU, 0};
  DataType dtype = kFloat;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  dropout_assign_float(input);

  Tensor output;
  double rate = 0.4;
  aitisa_dropout(input, rate, &output);

  float* output_data = (float*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  int64_t num_zero = 0;
  for (int64_t i = 0; i < output_size; i++) {
    if (output_data[i] == 0) {
      num_zero++;
    }
  }
  double actual_rate = (double)num_zero / (double)output_size;
  EXPECT_TRUE(abs(actual_rate - rate) < 0.2);

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api
