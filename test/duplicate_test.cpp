#include <math.h>
#include <vector>
#include "gtest/gtest.h"
extern "C" {
#include "src/basic/duplicate.h"
}

void duplicate_assign_float(Tensor t) {
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
TEST(Duplicate, Float) {
  Tensor input;
  int64_t dims[2] = {3, 2};
  Device device = {DEVICE_CPU, 0};
  DataType dtype = kFloat;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  duplicate_assign_float(input);
  Tensor output;
  aitisa_duplicate(input, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5};
  int64_t in_ndim = aitisa_tensor_ndim(input);
  int64_t out_ndim = aitisa_tensor_ndim(output);
  int64_t* in_dims = aitisa_tensor_dims(input);
  int64_t* out_dims = aitisa_tensor_dims(output);
  EXPECT_EQ(in_ndim, out_ndim);
  for (int64_t i = 0; i < in_ndim; i++) {
    EXPECT_EQ(in_dims[i], out_dims[i]);
  }
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_TRUE(out_data[i] == test_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api