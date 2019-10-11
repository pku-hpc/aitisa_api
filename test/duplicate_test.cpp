#include "gtest/gtest.h"
#include <math.h>
#include <vector> 
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
  int64_t dims[2] = { 3,2 };
  Device device = { DEVICE_CPU, 0 };
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 2, &input);
  duplicate_assign_float(input);
  Tensor output;
  aitisa_duplicate(input, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  std::vector<float> test_data = { 0, 0.1, 0.2, 0.3, 0.4, 0.5};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
		EXPECT_TRUE(out_data[i] == test_data[i]);
	}
}
}//namespace
}//namespace aitisa_api