#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/basic/reshape.h"
}

void reshape_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 4;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {
TEST(Reshape, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 1, 5, 10};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  reshape_assign_int32(input);
  int64_t in_size = aitisa_tensor_size(input);

  Tensor output;
  int64_t out_dims[3] = {2, 5, 10};
  int64_t out_ndim = 3;
  aitisa_reshape(input, out_dims, out_ndim, &output);
  int64_t out_size = aitisa_tensor_size(output);

  int64_t* test_out_dims = aitisa_tensor_dims(output);
  EXPECT_EQ(in_size, out_size);
  for (int64_t i = 0; i < out_ndim; i++) {
    EXPECT_EQ(test_out_dims[i], out_dims[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api