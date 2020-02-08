#include "gtest/gtest.h"
extern "C" {
#include "src/nn/sigmoid.h"
//#include "src/tool/tool.h"
}

void sigmoid_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Sigmoid, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  sigmoid_assign_float(input);
  int64_t size = aitisa_tensor_size(input);

  Tensor output;
  aitisa_sigmoid(input, &output);
  // tensor_printer2d(input);
  // tensor_printer2d(output);

  float test_data[] = {0.500000, 0.524979, 0.549834,
                       0.574443, 0.598688, 0.622459};
  float* out_data = (float*)aitisa_tensor_data(output);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_TRUE(abs(test_data[i] - out_data[i]) < 0.00001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api